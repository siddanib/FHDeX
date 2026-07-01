#ifndef IntermediateScatteringFunction_H_
#define IntermediateScatteringFunction_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <StructFact.H>

#include <deque>
#include <string>

class IntermediateScatteringFunction
{
public:
    void ReadParameters (amrex::ParmParse& pp);
    void Validate (int max_level, int alg_type, bool particles_enabled,
                   amrex::Geometry const& geom, int default_do_1D,
                   std::string const& default_mode);

    [[nodiscard]] bool Enabled () const;
    [[nodiscard]] bool SamplesThisStep (int step) const;

    void Sample (int step, amrex::Real time, amrex::MultiFab const& phi,
                 amrex::Geometry const& geom, amrex::Real dt);
    void Write (amrex::Geometry const& geom) const;

private:
    struct ActiveWindow {
        int start_step = 0;
        amrex::Vector<amrex::Real> anchor_real;
        amrex::Vector<amrex::Real> anchor_imag;
        amrex::Vector<amrex::Real> sample_real;
        amrex::Vector<amrex::Real> sample_imag;
        amrex::Vector<int> sample_seen;
    };

    struct SourceState {
        std::deque<ActiveWindow> active_windows;
        amrex::Vector<amrex::Real> accum_real;
        amrex::Vector<amrex::Real> accum_imag;
        amrex::Vector<int> accum_count;
    };

    void Init (amrex::MultiFab const& phi, amrex::Geometry const& geom, amrex::Real dt);
    void ExtractSourceModes (amrex::MultiFab const& density, amrex::Geometry const& geom,
                             amrex::Vector<amrex::Real>& real,
                             amrex::Vector<amrex::Real>& imag);
    void AccumulateSource (SourceState& state,
                           amrex::Vector<amrex::Real> const& current_real,
                           amrex::Vector<amrex::Real> const& current_imag,
                           int step, bool starts_window);

    int m_n_modes = 0;
    int m_q_min = 1;
    int m_q_stride = 1;
    int m_dt_meas = 0;
    int m_t_max = 0;
    int m_window_stride = 0;
    int m_relax_time = 0;
    std::string m_mode;
    std::string m_plot_file {"isf_density"};
    int m_do_1D = 0;
    bool m_use_spde = false;
    bool m_use_particle = false;
    bool m_initialized = false;
    int m_n_lags = 0;
    amrex::Real m_level_dt = amrex::Real(0.0);
    int m_npencils = 1;
    amrex::Real m_product_scaling = amrex::Real(1.0);
    amrex::Vector<int> m_q_indices;
    StructFact m_fft_full;
    StructFact m_fft_pencil;
    SourceState m_spde;
    SourceState m_particle;
};

#endif
