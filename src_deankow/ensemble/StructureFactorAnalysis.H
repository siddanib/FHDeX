#ifndef StructureFactorAnalysis_H_
#define StructureFactorAnalysis_H_

#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <StructFact.H>

#include <memory>
#include <string>

class StructureFactorAnalysis
{
public:
    void ReadParameters (amrex::ParmParse& pp);
    void Validate (int max_level, int alg_type, bool particles_enabled,
                   amrex::Geometry const& geom);

    [[nodiscard]] bool Enabled () const;
    [[nodiscard]] bool SamplesThisStep (int step) const;
    [[nodiscard]] bool WritesThisStep (int step, int plot_int) const;

    void Sample (amrex::MultiFab const& phi, amrex::Geometry const& geom);
    void Write (int step, amrex::Real time);

    [[nodiscard]] int Do1D () const noexcept { return m_do_1D; }
    [[nodiscard]] std::string const& Mode () const noexcept { return m_mode; }

private:
    void Init (amrex::MultiFab const& phi, amrex::Geometry const& geom);

    int m_struct_fact_int = -1;
    int m_n_steps_skip = 0;
    int m_do_1D = 0;
    std::string m_mode;
    std::string m_plot_file {"plt_SF_density"};
    int m_zero_avg = 1;
    bool m_use_spde = false;
    bool m_use_particle = false;
    bool m_initialized = false;
    int m_sample_count = 0;
    int m_npencils = 0;
    amrex::BoxArray m_pencil_ba;
    amrex::DistributionMapping m_pencil_dmap;
    StructFact m_spde;
    StructFact m_particle;
    amrex::Vector<std::unique_ptr<StructFact>> m_spde_pencils;
    amrex::Vector<std::unique_ptr<StructFact>> m_particle_pencils;
};

#endif
