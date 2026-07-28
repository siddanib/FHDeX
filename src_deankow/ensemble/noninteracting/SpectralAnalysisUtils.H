#ifndef SpectralAnalysisUtils_H_
#define SpectralAnalysisUtils_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>

#include <StructFact.H>

#include <string>

namespace SpectralAnalysis {

void CopyDensityComponent (amrex::MultiFab const& src, amrex::MultiFab& dst, int src_comp);

void ExtractXPencil (amrex::MultiFab const& mf, amrex::MultiFab& mf_pencil,
                     int pencily, int pencilz);

void WritePlotFilesSF1D (amrex::MultiFab const& mag, amrex::MultiFab const& realimag,
                         int step, amrex::Real time,
                         amrex::Vector<std::string> const& names,
                         std::string const& plotfile_base);

void ExtractSelectedXModes (StructFact& fft_tool,
                            amrex::MultiFab const& density,
                            amrex::Vector<int> const& q_indices,
                            amrex::Vector<amrex::Real>& mode_real,
                            amrex::Vector<amrex::Real>& mode_imag);

}

#endif
