#include <IntermediateScatteringFunction.H>

#include <SpectralAnalysisUtils.H>

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <utility>

void
IntermediateScatteringFunction::ReadParameters (amrex::ParmParse& pp)
{
    pp.query("isf_n_modes", m_n_modes);
    pp.query("isf_q_min", m_q_min);
    pp.query("isf_q_stride", m_q_stride);
    pp.query("isf_dt_meas", m_dt_meas);
    pp.query("isf_t_max", m_t_max);
    pp.query("isf_window_stride", m_window_stride);
    pp.query("isf_relax_time", m_relax_time);
    pp.query("isf_mode", m_mode);
    pp.query("isf_plot_file", m_plot_file);
}

bool
IntermediateScatteringFunction::Enabled () const
{
    return m_n_modes > 0;
}

bool
IntermediateScatteringFunction::SamplesThisStep (int /*step*/) const
{
    return Enabled();
}

void
IntermediateScatteringFunction::Validate (int max_level, int alg_type, bool particles_enabled,
                                          amrex::Geometry const& geom, int default_do_1D,
                                          std::string const& default_mode)
{
    m_use_spde = false;
    m_use_particle = false;
    m_do_1D = default_do_1D;

    if (!Enabled()) {
        return;
    }

    if (max_level != 0) {
        amrex::Abort("Intermediate structure-factor analysis currently requires amr.max_level = 0.");
    }
    if (m_do_1D != 0 && m_do_1D != 1) {
        amrex::Abort("do_1D must be 0 or 1 for intermediate structure-factor analysis.");
    }
    if (m_q_min < 1) {
        amrex::Abort("isf_q_min must be >= 1.");
    }
    if (m_q_stride < 1) {
        amrex::Abort("isf_q_stride must be >= 1.");
    }
    if (m_dt_meas <= 0) {
        amrex::Abort("isf_dt_meas must be > 0 time steps when ISF is enabled.");
    }
    if (m_t_max < 0) {
        amrex::Abort("isf_t_max must be >= 0 time steps when ISF is enabled.");
    }
    if (m_window_stride <= 0) {
        amrex::Abort("isf_window_stride must be > 0 time steps when ISF is enabled.");
    }
    if (m_relax_time < 0) {
        amrex::Abort("isf_relax_time must be >= 0 time steps when ISF is enabled.");
    }

    const int nx = geom.Domain().length(0);
    const int last_q = m_q_min + (m_n_modes - 1) * m_q_stride;
    if (last_q > nx / 2) {
        amrex::Abort("Requested ISF x Fourier mode exceeds nx/2.");
    }

    if (m_mode.empty()) {
        m_mode = default_mode.empty() ? (particles_enabled ? "both" : "spde") : default_mode;
    }

    if (m_mode == "spde") {
        m_use_spde = true;
    } else if (m_mode == "particle") {
        m_use_particle = true;
    } else if (m_mode == "both") {
        m_use_spde = true;
        m_use_particle = true;
    } else {
        amrex::Abort("isf_mode must be spde, particle, or both.");
    }

    if (m_use_particle) {
#ifndef AMREX_PARTICLES
        amrex::Abort("isf_mode=particle or both requires particle support.");
#else
        if (!particles_enabled) {
            amrex::Abort("isf_mode=particle or both requires amr.use_particles = 1.");
        }
#endif
        if (alg_type == 0) {
            amrex::Abort("isf_mode=particle or both requires alg_type != 0 so density component 1 is available.");
        }
    }
}

void
IntermediateScatteringFunction::Init (amrex::MultiFab const& phi, amrex::Geometry const& geom,
                                      amrex::Real dt)
{
    if (!Enabled() || m_initialized) {
        return;
    }

    if (phi.nComp() < 1) {
        amrex::Abort("SPDE density component 0 is unavailable for intermediate structure-factor analysis.");
    }
    if (m_use_particle && phi.nComp() < 2) {
        amrex::Abort("Particle density component 1 is unavailable for intermediate structure-factor analysis.");
    }

    m_level_dt = dt;
    if (m_dt_meas <= 0 || m_window_stride <= 0) {
        amrex::Abort("ISF measurement and window stride must be at least one time step.");
    }
    if (m_t_max % m_dt_meas != 0) {
        amrex::Abort("isf_t_max must be an integer multiple of isf_dt_meas.");
    }
    m_n_lags = m_t_max / m_dt_meas + 1;

    m_q_indices.resize(m_n_modes);
    for (int n = 0; n < m_n_modes; ++n) {
        m_q_indices[n] = m_q_min + n * m_q_stride;
    }

    const amrex::Real* dx = geom.CellSize();
    amrex::Real cell_volume = dx[0];
#if (AMREX_SPACEDIM > 1)
    cell_volume *= dx[1];
#endif
#if (AMREX_SPACEDIM > 2)
    cell_volume *= dx[2];
#endif
    m_product_scaling = cell_volume;

    amrex::Vector<std::string> names {"isf_rho"};
    amrex::Vector<amrex::Real> var_scaling {amrex::Real(1.0) / cell_volume};
    if (m_do_1D == 0) {
        m_npencils = 1;
        m_fft_full.define(phi.boxArray(), phi.DistributionMap(), names, var_scaling);
    } else {
        amrex::MultiFab sample_density;
        SpectralAnalysis::CopyDensityComponent(phi, sample_density, 0);
        amrex::MultiFab sample_pencil;
        SpectralAnalysis::ExtractXPencil(sample_density, sample_pencil,
                                         geom.Domain().smallEnd(AMREX_SPACEDIM > 1 ? 1 : 0),
                                         geom.Domain().smallEnd(AMREX_SPACEDIM > 2 ? 2 : 0));
        m_fft_pencil.define(sample_pencil.boxArray(), sample_pencil.DistributionMap(), names, var_scaling);

        const amrex::Box& domain = geom.Domain();
        m_npencils = 1;
#if (AMREX_SPACEDIM > 1)
        m_npencils *= domain.length(1);
#endif
#if (AMREX_SPACEDIM > 2)
        m_npencils *= domain.length(2);
#endif
    }

    auto init_source = [this] (SourceState& state)
    {
        state.active_windows.clear();
        state.accum_real.assign(m_n_lags * m_n_modes, amrex::Real(0.0));
        state.accum_imag.assign(m_n_lags * m_n_modes, amrex::Real(0.0));
        state.accum_count.assign(m_n_lags, 0);
    };
    if (m_use_spde) { init_source(m_spde); }
    if (m_use_particle) { init_source(m_particle); }

    m_initialized = true;
}

void
IntermediateScatteringFunction::ExtractSourceModes (amrex::MultiFab const& density,
                                                    amrex::Geometry const& geom,
                                                    amrex::Vector<amrex::Real>& real,
                                                    amrex::Vector<amrex::Real>& imag)
{
    real.assign(m_npencils * m_n_modes, amrex::Real(0.0));
    imag.assign(m_npencils * m_n_modes, amrex::Real(0.0));
    if (m_do_1D == 0) {
        SpectralAnalysis::ExtractSelectedXModes(m_fft_full, density, m_q_indices, real, imag);
        return;
    }

    const amrex::Box& domain = geom.Domain();
#if (AMREX_SPACEDIM > 1)
    const int ylo = domain.smallEnd(1);
    const int ny = domain.length(1);
#else
    const int ylo = 0;
    const int ny = 1;
#endif
#if (AMREX_SPACEDIM > 2)
    const int zlo = domain.smallEnd(2);
#else
    const int zlo = 0;
#endif

    for (int p = 0; p < m_npencils; ++p) {
        const int pencily = ylo + p % ny;
        const int pencilz = zlo + p / ny;
        amrex::MultiFab pencil;
        SpectralAnalysis::ExtractXPencil(density, pencil, pencily, pencilz);
        amrex::Vector<amrex::Real> pencil_real;
        amrex::Vector<amrex::Real> pencil_imag;
        SpectralAnalysis::ExtractSelectedXModes(m_fft_pencil, pencil, m_q_indices,
                                                pencil_real, pencil_imag);
        for (int m = 0; m < m_n_modes; ++m) {
            const int idx = p * m_n_modes + m;
            real[idx] = pencil_real[m];
            imag[idx] = pencil_imag[m];
        }
    }
}

void
IntermediateScatteringFunction::AccumulateSource (
    SourceState& state,
    amrex::Vector<amrex::Real> const& current_real,
    amrex::Vector<amrex::Real> const& current_imag,
    int step,
    bool starts_window)
{
    if (starts_window) {
        ActiveWindow window;
        window.start_step = step;
        window.anchor_real = current_real;
        window.anchor_imag = current_imag;
        window.sample_real.assign(m_n_lags * m_n_modes, amrex::Real(0.0));
        window.sample_imag.assign(m_n_lags * m_n_modes, amrex::Real(0.0));
        window.sample_seen.assign(m_n_lags, 0);
        state.active_windows.push_back(std::move(window));
    }

    for (auto& window : state.active_windows) {
        const int delta = step - window.start_step;
        if (delta < 0 || delta > m_t_max || delta % m_dt_meas != 0) {
            continue;
        }
        const int lag = delta / m_dt_meas;
        const amrex::Real inv_npencils = amrex::Real(1.0) / static_cast<amrex::Real>(m_npencils);
        for (int m = 0; m < m_n_modes; ++m) {
            amrex::Real prod_real = amrex::Real(0.0);
            amrex::Real prod_imag = amrex::Real(0.0);
            for (int p = 0; p < m_npencils; ++p) {
                const int idx = p * m_n_modes + m;
                const amrex::Real ar = window.anchor_real[idx];
                const amrex::Real ai = window.anchor_imag[idx];
                const amrex::Real cr = current_real[idx];
                const amrex::Real ci = current_imag[idx];
                prod_real += ar * cr + ai * ci;
                prod_imag += ai * cr - ar * ci;
            }
            const int out = lag * m_n_modes + m;
            window.sample_real[out] = m_product_scaling * inv_npencils * prod_real;
            window.sample_imag[out] = m_product_scaling * inv_npencils * prod_imag;
        }
        window.sample_seen[lag] = 1;

        if (delta == m_t_max) {
            bool complete = true;
            for (int seen : window.sample_seen) {
                if (seen == 0) {
                    complete = false;
                    break;
                }
            }
            if (complete) {
                for (int l = 0; l < m_n_lags; ++l) {
                    for (int m = 0; m < m_n_modes; ++m) {
                        const int out = l * m_n_modes + m;
                        state.accum_real[out] += window.sample_real[out];
                        state.accum_imag[out] += window.sample_imag[out];
                    }
                    state.accum_count[l] += 1;
                }
            }
        }
    }

    while (!state.active_windows.empty() &&
           step - state.active_windows.front().start_step >= m_t_max) {
        state.active_windows.pop_front();
    }
}

void
IntermediateScatteringFunction::Sample (int step, amrex::Real time,
                                        amrex::MultiFab const& phi,
                                        amrex::Geometry const& geom,
                                        amrex::Real dt)
{
    amrex::ignore_unused(time);
    Init(phi, geom, dt);
    if (!m_initialized || step < m_relax_time) {
        return;
    }

    const bool starts_window = ((step - m_relax_time) % m_window_stride) == 0;
    auto source_due = [this, starts_window, step] (SourceState const& state) -> bool
    {
        if (starts_window) {
            return true;
        }
        for (auto const& window : state.active_windows) {
            const int delta = step - window.start_step;
            if (delta >= 0 && delta <= m_t_max && delta % m_dt_meas == 0) {
                return true;
            }
        }
        return false;
    };

    const bool need_spde = m_use_spde && source_due(m_spde);
    const bool need_particle = m_use_particle && source_due(m_particle);
    if (!need_spde && !need_particle) {
        return;
    }

    if (need_spde) {
        amrex::MultiFab density;
        SpectralAnalysis::CopyDensityComponent(phi, density, 0);
        amrex::Vector<amrex::Real> current_real;
        amrex::Vector<amrex::Real> current_imag;
        ExtractSourceModes(density, geom, current_real, current_imag);
        AccumulateSource(m_spde, current_real, current_imag, step, starts_window);
    }

    if (need_particle) {
        amrex::MultiFab density;
        SpectralAnalysis::CopyDensityComponent(phi, density, 1);
        amrex::Vector<amrex::Real> current_real;
        amrex::Vector<amrex::Real> current_imag;
        ExtractSourceModes(density, geom, current_real, current_imag);
        AccumulateSource(m_particle, current_real, current_imag, step, starts_window);
    }
}

void
IntermediateScatteringFunction::Write (amrex::Geometry const& geom) const
{
    if (!m_initialized) {
        return;
    }

    auto write_source = [&] (SourceState const& state, std::string const& source_name)
    {
        bool have_samples = false;
        for (int count : state.accum_count) {
            if (count > 0) {
                have_samples = true;
                break;
            }
        }
        if (!have_samples || !amrex::ParallelDescriptor::IOProcessor()) {
            return;
        }

        const amrex::Real lx = geom.ProbLength(0);
        for (int m = 0; m < m_n_modes; ++m) {
            const int q = m_q_indices[m];
            const amrex::Real kx = amrex::Real(6.283185307179586476925286766559005768L) * static_cast<amrex::Real>(q) / lx;
            std::string filename = m_plot_file + "_" + source_name + (m_do_1D ? "_1D" : "") + "_q" + std::to_string(q) + ".dat";
            std::ofstream os(filename);
            if (!os.good()) {
                amrex::FileOpenFailed(filename);
            }
            os << "# lag_index lag_time q_index kx F_real F_imag F_mag n_windows\n";
            os << std::setprecision(17);
            for (int lag = 0; lag < m_n_lags; ++lag) {
                const int count = state.accum_count[lag];
                if (count <= 0) {
                    continue;
                }
                const amrex::Real lag_time = static_cast<amrex::Real>(lag * m_dt_meas) * m_level_dt;
                const int idx = lag * m_n_modes + m;
                const amrex::Real freal = state.accum_real[idx] / static_cast<amrex::Real>(count);
                const amrex::Real fimag = state.accum_imag[idx] / static_cast<amrex::Real>(count);
                const amrex::Real fmag = std::sqrt(freal * freal + fimag * fimag);
                os << lag << " " << lag_time << " "
                   << q << " " << kx << " "
                   << freal << " " << fimag << " " << fmag << " " << count << "\n";
            }
        }
    };

    if (m_use_spde) {
        write_source(m_spde, "spde_rho");
    }
    if (m_use_particle) {
        write_source(m_particle, "particle_rho");
    }
}
