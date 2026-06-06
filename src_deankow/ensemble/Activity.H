#ifndef ACTIVITY_H_
#define ACTIVITY_H_

#include <AMReX_Extension.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <string>

enum class ActivityType {
    none = 0,
    rtp,
    abp,
    aoup
};

struct Activity {
    bool enabled = false;
    ActivityType type = ActivityType::none;
    amrex::Real D_trans = amrex::Real(0.5);
    amrex::Real persistence_length = amrex::Real(0.0);
    amrex::Real persistence_time = amrex::Real(0.0);
};

inline std::string ActivityTypeToString (ActivityType type)
{
    switch (type) {
    case ActivityType::none:
        return "none";
    case ActivityType::rtp:
        return "rtp";
    case ActivityType::abp:
        return "abp";
    case ActivityType::aoup:
        return "aoup";
    }
    return "unknown";
}

inline ActivityType ActivityTypeFromString (std::string const& type_name)
{
    if (type_name == "rtp") {
        return ActivityType::rtp;
    }
    if (type_name == "abp") {
        return ActivityType::abp;
    }
    if (type_name == "aoup") {
        return ActivityType::aoup;
    }
    if (type_name == "none") {
        return ActivityType::none;
    }

    return static_cast<ActivityType>(-1);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void active_push (Activity const& activity,
                  EnsembleDirection const& ensemble_dir,
                  amrex::Real const& dt,
                  amrex::Real& incx,
                  amrex::Real& incy,
                  amrex::Real& incz,
                  amrex::Real& orientx,
                  amrex::Real& orienty,
                  amrex::Real& orientz,
                  amrex::RandomEngine const& engine) noexcept
{

    if (!activity.enabled || activity.type == ActivityType::none) {
        incx = 0.0;
        incy = 0.0;
        incz = 0.0;
    }

    if (activity.type == ActivityType::rtp) {
        
        amrex::Real alpha = amrex::Real(1.0)/activity.persistence_time;
        amrex::Real v_persist = alpha*activity.persistence_length;

        bool do_1D;

        if (ensemble_dir[1] != 0) do_1D = true;

        if (do_1D) {
            incy = 0.0;
            incz = 0.0;

            amrex::Real cur_dir;
            if (std::signbit(orientx)) cur_dir = amrex::Real(-1.0);
            else cur_dir = amrex::Real(1.0);

            incx = cur_dir * v_persist * dt;

            // for next step
            if (amrex::Random(engine) < 0.5*alpha*dt) orientx = amrex::Real(-1.0)*cur_dir;
            else orientx = amrex::Real(1.0)*cur_dir;
        }
        else {
            incz = 0.0;
            incx = orientx * v_persist * dt;
            incy = orienty * v_persist * dt;

            // for next step
            if (amrex::Random(engine) < alpha*dt) {
                amrex::Real orientx_tmp = amrex::Real(-1.0) + amrex::Real(2.0)*amrex::Random(engine);
                amrex::Real orienty_tmp = amrex::Real(-1.0) + amrex::Real(2.0)*amrex::Random(engine);
                amrex::Real mag = std::sqrt(orientx_tmp*orientx_tmp + orienty_tmp*orienty_tmp);
                orientx = orientx_tmp/mag;
                orienty = orienty_tmp/mag;
            }
        }
    }
}

#endif

