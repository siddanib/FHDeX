#include <StructureFactorAnalysis.H>

#include <SpectralAnalysisUtils.H>

#include <AMReX.H>

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
        amrex::MultiFab sample_density;
        SpectralAnalysis::CopyDensityComponent(phi, sample_density, 0);
        amrex::MultiFab sample_pencil;
        SpectralAnalysis::ExtractXPencil(sample_density, sample_pencil,
                                         geom.Domain().smallEnd(AMREX_SPACEDIM > 1 ? 1 : 0),
                                         geom.Domain().smallEnd(AMREX_SPACEDIM > 2 ? 2 : 0));
        m_pencil_ba = sample_pencil.boxArray();
        m_pencil_dmap = sample_pencil.DistributionMap();

        const amrex::Box& domain = geom.Domain();
        m_npencils = 1;
#if (AMREX_SPACEDIM > 1)
        m_npencils *= domain.length(1);
#endif
#if (AMREX_SPACEDIM > 2)
        m_npencils *= domain.length(2);
#endif

        if (m_use_spde) {
            m_spde_pencils.resize(m_npencils);
            for (int i = 0; i < m_npencils; ++i) {
                m_spde_pencils[i] = std::make_unique<StructFact>();
                m_spde_pencils[i]->define(m_pencil_ba, m_pencil_dmap, spde_names, var_scaling);
            }
        }
        if (m_use_particle) {
            m_particle_pencils.resize(m_npencils);
            for (int i = 0; i < m_npencils; ++i) {
                m_particle_pencils[i] = std::make_unique<StructFact>();
                m_particle_pencils[i]->define(m_pencil_ba, m_pencil_dmap, particle_names, var_scaling);
            }
        }
    }

    m_initialized = true;
}

void
StructureFactorAnalysis::Sample (amrex::MultiFab const& phi, amrex::Geometry const& geom)
{
    Init(phi, geom);

    amrex::MultiFab spde_density;
    amrex::MultiFab particle_density;
    if (m_use_spde) {
        SpectralAnalysis::CopyDensityComponent(phi, spde_density, 0);
    }
    if (m_use_particle) {
        SpectralAnalysis::CopyDensityComponent(phi, particle_density, 1);
    }

    if (m_do_1D == 0) {
        if (m_use_spde) { m_spde.FortStructure(spde_density); }
        if (m_use_particle) { m_particle.FortStructure(particle_density); }
    } else {
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
            if (m_use_spde) {
                amrex::MultiFab pencil;
                SpectralAnalysis::ExtractXPencil(spde_density, pencil, pencily, pencilz);
                m_spde_pencils[p]->FortStructure(pencil);
            }
            if (m_use_particle) {
                amrex::MultiFab pencil;
                SpectralAnalysis::ExtractXPencil(particle_density, pencil, pencily, pencilz);
                m_particle_pencils[p]->FortStructure(pencil);
            }
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

    auto write_averaged = [&] (amrex::Vector<std::unique_ptr<StructFact>>& pencils,
                               std::string const& base)
    {
        if (pencils.empty()) {
            return;
        }

        amrex::MultiFab mag(m_pencil_ba, m_pencil_dmap, pencils[0]->get_ncov(), 0);
        amrex::MultiFab realimag(m_pencil_ba, m_pencil_dmap, 2 * pencils[0]->get_ncov(), 0);
        mag.setVal(amrex::Real(0.0));
        realimag.setVal(amrex::Real(0.0));

        for (auto& sf : pencils) {
            sf->AddToExternal(mag, realimag, m_zero_avg);
        }

        const amrex::Real inv_npencils = amrex::Real(1.0) / static_cast<amrex::Real>(m_npencils);
        mag.mult(inv_npencils);
        realimag.mult(inv_npencils);

        SpectralAnalysis::WritePlotFilesSF1D(mag, realimag, step, time, pencils[0]->get_names(), base);
    };

    if (m_use_spde) {
        write_averaged(m_spde_pencils, spde_base);
    }
    if (m_use_particle) {
        write_averaged(m_particle_pencils, particle_base);
    }
}
