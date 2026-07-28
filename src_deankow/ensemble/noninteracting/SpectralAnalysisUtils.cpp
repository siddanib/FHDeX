#include <SpectralAnalysisUtils.H>

#include <AMReX_GpuContainers.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>

namespace SpectralAnalysis {

void
CopyDensityComponent (amrex::MultiFab const& src, amrex::MultiFab& dst, int src_comp)
{
    dst.define(src.boxArray(), src.DistributionMap(), 1, 0);
    amrex::MultiFab::Copy(dst, src, src_comp, 0, 1, 0);
}

void
ExtractXPencil (amrex::MultiFab const& mf, amrex::MultiFab& mf_pencil,
                int pencily, int pencilz)
{
    amrex::Box domain(mf.boxArray().minimalBox());
    amrex::IntVect dom_lo(domain.loVect());
    amrex::IntVect dom_hi(domain.hiVect());

#if (AMREX_SPACEDIM > 1)
    dom_lo[1] = dom_hi[1] = pencily;
#else
    amrex::ignore_unused(pencily);
#endif
#if (AMREX_SPACEDIM > 2)
    dom_lo[2] = dom_hi[2] = pencilz;
#else
    amrex::ignore_unused(pencilz);
#endif

    amrex::Box domain_pencil(dom_lo, dom_hi);
    amrex::BoxArray ba_pencil(domain_pencil);
    amrex::DistributionMapping dmap_pencil(ba_pencil);
    amrex::MultiFab mf_pencil_tmp(ba_pencil, dmap_pencil, 1, 0);
    mf_pencil_tmp.ParallelCopy(mf, 0, 0, 1);

#if (AMREX_SPACEDIM > 1)
    dom_lo[1] = dom_hi[1] = 0;
#endif
#if (AMREX_SPACEDIM > 2)
    dom_lo[2] = dom_hi[2] = 0;
#endif

    amrex::Box domain_pencil_zeroed(dom_lo, dom_hi);
    amrex::BoxArray ba_pencil_zeroed(domain_pencil_zeroed);
    mf_pencil.define(ba_pencil_zeroed, dmap_pencil, 1, 0);

    for (amrex::MFIter mfi(mf_pencil_tmp, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.tilebox();
        auto const& pencil = mf_pencil.array(mfi);
        auto const& pencil_tmp = mf_pencil_tmp.const_array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            pencil(i, 0, 0) = pencil_tmp(i, j, k);
        });
    }
}

namespace {

amrex::Geometry
MakeXPencilSFGeometry (amrex::Box const& domain_pencil)
{
    amrex::Vector<int> is_periodic(AMREX_SPACEDIM, 1);
    amrex::Vector<amrex::Real> projected_lo(AMREX_SPACEDIM, amrex::Real(-0.5));
    amrex::Vector<amrex::Real> projected_hi(AMREX_SPACEDIM, amrex::Real(0.5));

    projected_lo[0] = -domain_pencil.length(0) / 2 - amrex::Real(0.5);
    projected_hi[0] =  domain_pencil.length(0) / 2 - amrex::Real(1.0) + amrex::Real(0.5);

    amrex::RealBox real_box_pencil({AMREX_D_DECL(projected_lo[0], projected_lo[1], projected_lo[2])},
                                   {AMREX_D_DECL(projected_hi[0], projected_hi[1], projected_hi[2])});

    amrex::Geometry geom;
    geom.define(domain_pencil, &real_box_pencil, amrex::CoordSys::cartesian, is_periodic.data());
    return geom;
}

}

void
WritePlotFilesSF1D (amrex::MultiFab const& mag, amrex::MultiFab const& realimag,
                    int step, amrex::Real time,
                    amrex::Vector<std::string> const& names,
                    std::string const& plotfile_base)
{
    const amrex::Geometry geom = MakeXPencilSFGeometry(mag.boxArray().minimalBox());

    amrex::Vector<std::string> var_names(names.size());
    for (int n = 0; n < names.size(); ++n) {
        var_names[n] = names[n];
    }

    std::string name = plotfile_base + "_mag";
    amrex::WriteSingleLevelPlotfile(amrex::Concatenate(name, step, 9),
                                    mag, var_names, geom, time, step);

    var_names.resize(2 * names.size());
    int cnt = 0;
    for (int n = 0; n < names.size(); ++n) {
        var_names[cnt] = names[n] + "_real";
        ++cnt;
    }
    for (int n = 0; n < names.size(); ++n) {
        var_names[cnt] = names[n] + "_imag";
        ++cnt;
    }

    name = plotfile_base + "_real_imag";
    amrex::WriteSingleLevelPlotfile(amrex::Concatenate(name, step, 9),
                                    realimag, var_names, geom, time, step);
}

void
ExtractSelectedXModes (StructFact& fft_tool,
                       amrex::MultiFab const& density,
                       amrex::Vector<int> const& q_indices,
                       amrex::Vector<amrex::Real>& mode_real,
                       amrex::Vector<amrex::Real>& mode_imag)
{
    const int nmodes = static_cast<int>(q_indices.size());
    mode_real.assign(nmodes, amrex::Real(0.0));
    mode_imag.assign(nmodes, amrex::Real(0.0));
    if (nmodes == 0) {
        return;
    }

    amrex::MultiFab dft_real(density.boxArray(), density.DistributionMap(), 1, 0);
    amrex::MultiFab dft_imag(density.boxArray(), density.DistributionMap(), 1, 0);
    fft_tool.ComputeFFT(density, dft_real, dft_imag);

    amrex::Gpu::DeviceVector<int> q_device(nmodes);
    amrex::Gpu::DeviceVector<amrex::Real> real_device(nmodes, amrex::Real(0.0));
    amrex::Gpu::DeviceVector<amrex::Real> imag_device(nmodes, amrex::Real(0.0));
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, q_indices.begin(), q_indices.end(), q_device.begin());

    int const* q_ptr = q_device.dataPtr();
    amrex::Real* real_ptr = real_device.dataPtr();
    amrex::Real* imag_ptr = imag_device.dataPtr();

    for (amrex::MFIter mfi(dft_real, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const amrex::Box& bx = mfi.validbox();
        const int xlo = bx.smallEnd(0);
        const int xhi = bx.bigEnd(0);
#if (AMREX_SPACEDIM > 1)
        const bool has_y0 = bx.smallEnd(1) <= 0 && bx.bigEnd(1) >= 0;
#else
        const bool has_y0 = true;
#endif
#if (AMREX_SPACEDIM > 2)
        const bool has_z0 = bx.smallEnd(2) <= 0 && bx.bigEnd(2) >= 0;
#else
        const bool has_z0 = true;
#endif
        if (!has_y0 || !has_z0) {
            continue;
        }

        auto const& real_arr = dft_real.const_array(mfi);
        auto const& imag_arr = dft_imag.const_array(mfi);
        amrex::ParallelFor(nmodes, [=] AMREX_GPU_DEVICE (int n) noexcept
        {
            const int q = q_ptr[n];
            if (q >= xlo && q <= xhi) {
                real_ptr[n] = real_arr(q, 0, 0, 0);
                imag_ptr[n] = imag_arr(q, 0, 0, 0);
            }
        });
    }

    amrex::Gpu::streamSynchronize();
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, real_device.begin(), real_device.end(), mode_real.begin());
    amrex::Gpu::copy(amrex::Gpu::deviceToHost, imag_device.begin(), imag_device.end(), mode_imag.begin());
    amrex::ParallelDescriptor::ReduceRealSum(mode_real.data(), nmodes);
    amrex::ParallelDescriptor::ReduceRealSum(mode_imag.data(), nmodes);
}

}
