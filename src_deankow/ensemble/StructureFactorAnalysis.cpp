#include <StructureFactorAnalysis.H>

#include <SpectralAnalysisUtils.H>

#include <AMReX.H>
#include <AMReX_FFT.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>

struct StructureFactorBatchedPencilSF
{
    using FFT = amrex::FFT::R2C<amrex::Real, amrex::FFT::Direction::forward>;
    using SpectralMF = amrex::FabArray<amrex::BaseFab<amrex::GpuComplex<amrex::Real>>>;

    int nx = 0;
    int npencils = 1;
    amrex::Real cell_volume = amrex::Real(1.0);
    amrex::Vector<std::string> names;
    std::unique_ptr<FFT> fft;
    SpectralMF spectral;
    amrex::Gpu::DeviceVector<amrex::Real> mode_sum;
    amrex::Gpu::DeviceVector<amrex::Real> plot_values;
    amrex::Vector<amrex::Real> host_mode_sum;
    amrex::Vector<amrex::Real> running_sum;

    void define (amrex::Geometry const& geom, std::string const& var_name,
                 amrex::Real cell_volume_in, int npencils_in)
    {
        nx = geom.Domain().length(0);
        npencils = npencils_in;
        cell_volume = cell_volume_in;
        names = {"struct_fact_" + var_name + "_" + var_name};

        amrex::FFT::Info info;
        info.setOneDMode(true);
        fft = std::make_unique<FFT>(geom.Domain(), info);
        auto layout = fft->getSpectralDataLayout();
        spectral.define(layout.first, layout.second, 1, 0);

        mode_sum.resize(nx);
        plot_values.resize(nx);
        host_mode_sum.assign(nx, amrex::Real(0.0));
        running_sum.assign(nx, amrex::Real(0.0));
    }

    void sample (amrex::MultiFab const& phi, int src_comp)
    {
        BL_PROFILE("StructureFactorBatchedPencilSF::sample");

        amrex::Real* mode_sum_ptr = mode_sum.dataPtr();
        amrex::ParallelFor(nx, [=] AMREX_GPU_DEVICE (int i) noexcept
        {
            mode_sum_ptr[i] = amrex::Real(0.0);
        });

        fft->forward(phi, spectral, src_comp, 0);

        const int nx_local = nx;
        const amrex::Real inv_nx = amrex::Real(1.0) / static_cast<amrex::Real>(nx);
        for (amrex::MFIter mfi(spectral, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto const& spec = spectral.const_array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const auto z = spec(i,j,k,0);
                const amrex::Real power = (z.real()*z.real() + z.imag()*z.imag()) * inv_nx;
                amrex::Gpu::Atomic::AddNoRet(&mode_sum_ptr[i], power);

                const int mirror = nx_local - i;
                if (i > 0 && mirror != i) {
                    amrex::Gpu::Atomic::AddNoRet(&mode_sum_ptr[mirror], power);
                }
            });
        }

        amrex::Gpu::streamSynchronize();
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, mode_sum.begin(), mode_sum.end(),
                         host_mode_sum.begin());
        amrex::Gpu::streamSynchronize();
        amrex::ParallelDescriptor::ReduceRealSum(host_mode_sum.data(), nx);

        for (int i = 0; i < nx; ++i) {
            running_sum[i] += host_mode_sum[i];
        }
    }

    void write (amrex::BoxArray const& ba, amrex::DistributionMapping const& dm,
                int sample_count, int zero_avg, int step, amrex::Real time,
                std::string const& base)
    {
        BL_PROFILE("StructureFactorBatchedPencilSF::write");

        amrex::MultiFab mag(ba, dm, 1, 0);
        amrex::MultiFab realimag(ba, dm, 2, 0);
        mag.setVal(amrex::Real(0.0));
        realimag.setVal(amrex::Real(0.0));

        amrex::Vector<amrex::Real> shifted(nx, amrex::Real(0.0));
        const int nxh = nx / 2;
        const amrex::Real scale = cell_volume /
            (static_cast<amrex::Real>(sample_count) * static_cast<amrex::Real>(npencils));
        for (int out = 0; out < nx; ++out) {
            const int src = (out - nxh + nx) % nx;
            shifted[out] = (zero_avg == 1 && src == 0) ? amrex::Real(0.0) : running_sum[src] * scale;
        }

        amrex::Gpu::copy(amrex::Gpu::hostToDevice, shifted.begin(), shifted.end(),
                         plot_values.begin());
        amrex::Real const* plot_ptr = plot_values.dataPtr();
        const int xlo = ba.minimalBox().smallEnd(0);

        for (amrex::MFIter mfi(mag, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
            const amrex::Box& bx = mfi.tilebox();
            auto const& mag_arr = mag.array(mfi);
            auto const& realimag_arr = realimag.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const amrex::Real value = plot_ptr[i - xlo];
                mag_arr(i,j,k,0) = value >= amrex::Real(0.0) ? value : -value;
                realimag_arr(i,j,k,0) = value;
                realimag_arr(i,j,k,1) = amrex::Real(0.0);
            });
        }
        amrex::Gpu::streamSynchronize();

        SpectralAnalysis::WritePlotFilesSF1D(mag, realimag, step, time, names, base);
    }
};

StructureFactorAnalysis::StructureFactorAnalysis() = default;

StructureFactorAnalysis::~StructureFactorAnalysis() = default;

void
StructureFactorAnalysis::ReadParameters (amrex::ParmParse& pp)
{
    int legacy_struct_fact_int = m_struct_fact_int;
    const bool have_struct_fact_int = pp.query("struct_fact_int", m_struct_fact_int);
    if (!have_struct_fact_int && pp.query("struc_fact_int", legacy_struct_fact_int)) {
        m_struct_fact_int = legacy_struct_fact_int;
    }
    pp.query("n_steps_skip", m_n_steps_skip);
    pp.query("sf_mode", m_mode);
    pp.query("do_1D", m_do_1D);
    pp.query("sf_plot_file", m_plot_file);
    pp.query("sf_zero_avg", m_zero_avg);
}

bool
StructureFactorAnalysis::Enabled () const
{
    return m_struct_fact_int > 0;
}

bool
StructureFactorAnalysis::SamplesThisStep (int step) const
{
    return Enabled() &&
           step > m_n_steps_skip &&
           step % m_struct_fact_int == 0;
}

bool
StructureFactorAnalysis::WritesThisStep (int step, int plot_int) const
{
    return Enabled() &&
           m_sample_count > 0 &&
           plot_int > 0 &&
           step > m_n_steps_skip &&
           step % plot_int == 0;
}

void
StructureFactorAnalysis::Validate (int max_level, int alg_type, bool particles_enabled,
                                   amrex::Geometry const& geom)
{
    m_use_spde = false;
    m_use_particle = false;

    if (!Enabled()) {
        return;
    }

    if (max_level != 0) {
        amrex::Abort("Structure-factor analysis currently requires amr.max_level = 0.");
    }
    if (m_do_1D != 0 && m_do_1D != 1) {
        amrex::Abort("do_1D must be 0 or 1 for structure-factor analysis.");
    }
    if (m_n_steps_skip < 0) {
        amrex::Abort("n_steps_skip must be >= 0 for ensemble structure-factor accumulation.");
    }
    if (m_zero_avg != 0 && m_zero_avg != 1) {
        amrex::Abort("sf_zero_avg must be 0 or 1.");
    }
    if (m_do_1D == 1 && geom.Domain().length(0) <= 1) {
        amrex::Abort("do_1D structure-factor analysis requires more than one x cell.");
    }

    if (m_mode.empty()) {
        m_mode = particles_enabled ? "both" : "spde";
    }

    if (m_mode == "spde") {
        m_use_spde = true;
    } else if (m_mode == "particle") {
        m_use_particle = true;
    } else if (m_mode == "both") {
        m_use_spde = true;
        m_use_particle = true;
    } else {
        amrex::Abort("sf_mode must be spde, particle, or both.");
    }

    if (m_use_particle) {
#ifndef AMREX_PARTICLES
        amrex::Abort("sf_mode=particle or both requires particle support.");
#else
        if (!particles_enabled) {
            amrex::Abort("sf_mode=particle or both requires amr.use_particles = 1.");
        }
#endif
        if (alg_type == 0) {
            amrex::Abort("sf_mode=particle or both requires alg_type != 0 so density component 1 is available.");
        }
    }
}

void
StructureFactorAnalysis::Init (amrex::MultiFab const& phi, amrex::Geometry const& geom)
{
    if (!Enabled() || m_initialized) {
        return;
    }

    if (phi.nComp() < 1) {
        amrex::Abort("SPDE density component 0 is unavailable for structure-factor analysis.");
    }
    if (m_use_particle && phi.nComp() < 2) {
        amrex::Abort("Particle density component 1 is unavailable for structure-factor analysis.");
    }

    const amrex::Real* dx = geom.CellSize();
    amrex::Real cell_volume = dx[0];
#if (AMREX_SPACEDIM > 1)
    cell_volume *= dx[1];
#endif
#if (AMREX_SPACEDIM > 2)
    cell_volume *= dx[2];
#endif

    amrex::Vector<std::string> spde_names {"spde_rho"};
    amrex::Vector<std::string> particle_names {"particle_rho"};
    amrex::Vector<amrex::Real> var_scaling {amrex::Real(1.0) / cell_volume};

    if (m_do_1D == 0) {
        if (m_use_spde) {
            m_spde.define(phi.boxArray(), phi.DistributionMap(), spde_names, var_scaling);
        }
        if (m_use_particle) {
            m_particle.define(phi.boxArray(), phi.DistributionMap(), particle_names, var_scaling);
        }
    } else {
        const amrex::Box& domain = geom.Domain();
        amrex::IntVect pencil_lo(domain.loVect());
        amrex::IntVect pencil_hi(domain.hiVect());
#if (AMREX_SPACEDIM > 1)
        pencil_lo[1] = pencil_hi[1] = 0;
#endif
#if (AMREX_SPACEDIM > 2)
        pencil_lo[2] = pencil_hi[2] = 0;
#endif
        m_pencil_ba.define(amrex::Box(pencil_lo, pencil_hi));
        m_pencil_dmap = amrex::DistributionMapping(m_pencil_ba);

        m_npencils = 1;
#if (AMREX_SPACEDIM > 1)
        m_npencils *= domain.length(1);
#endif
#if (AMREX_SPACEDIM > 2)
        m_npencils *= domain.length(2);
#endif

        if (m_use_spde) {
            m_spde_pencils = std::make_unique<StructureFactorBatchedPencilSF>();
            m_spde_pencils->define(geom, spde_names[0], cell_volume, m_npencils);
        }
        if (m_use_particle) {
            m_particle_pencils = std::make_unique<StructureFactorBatchedPencilSF>();
            m_particle_pencils->define(geom, particle_names[0], cell_volume, m_npencils);
        }
    }

    m_initialized = true;
}

void
StructureFactorAnalysis::Sample (amrex::MultiFab const& phi, amrex::Geometry const& geom)
{
    Init(phi, geom);

    if (m_do_1D == 0) {
        amrex::MultiFab spde_density;
        amrex::MultiFab particle_density;
        if (m_use_spde) {
            SpectralAnalysis::CopyDensityComponent(phi, spde_density, 0);
            m_spde.FortStructure(spde_density);
        }
        if (m_use_particle) {
            SpectralAnalysis::CopyDensityComponent(phi, particle_density, 1);
            m_particle.FortStructure(particle_density);
        }
    } else {
        if (m_use_spde) {
            m_spde_pencils->sample(phi, 0);
        }
        if (m_use_particle) {
            m_particle_pencils->sample(phi, 1);
        }
    }

    ++m_sample_count;
}

void
StructureFactorAnalysis::Write (int step, amrex::Real time)
{
    if (!m_initialized || m_sample_count == 0) {
        return;
    }

    const std::string spde_base = m_plot_file + "_spde_rho" + (m_do_1D ? "_1D" : "");
    const std::string particle_base = m_plot_file + "_particle_rho" + (m_do_1D ? "_1D" : "");

    if (m_do_1D == 0) {
        if (m_use_spde) {
            m_spde.WritePlotFile(step, time, spde_base, m_zero_avg);
        }
        if (m_use_particle) {
            m_particle.WritePlotFile(step, time, particle_base, m_zero_avg);
        }
        return;
    }

    if (m_use_spde) {
        m_spde_pencils->write(m_pencil_ba, m_pencil_dmap, m_sample_count,
                              m_zero_avg, step, time, spde_base);
    }
    if (m_use_particle) {
        m_particle_pencils->write(m_pencil_ba, m_pencil_dmap, m_sample_count,
                                  m_zero_avg, step, time, particle_base);
    }
}
