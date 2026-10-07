#include <librealsense2/rs.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

int main()
try
{
    rs2::context context;
    auto devices = context.query_devices();
    if (devices.size() == 0)
        throw std::runtime_error("No RealSense device was found");

    auto device = devices.front();
    std::cout << "Device: " << device.get_info(RS2_CAMERA_INFO_NAME) << '\n'
              << "Serial: " << device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << '\n'
              << "Firmware: " << device.get_info(RS2_CAMERA_INFO_FIRMWARE_VERSION) << '\n';

    rs2::pipeline pipeline(context);
    rs2::config configuration;
    configuration.enable_stream(RS2_STREAM_DEPTH, 640, 0, RS2_FORMAT_Z16, 30);
    pipeline.start(configuration);

    std::uint64_t valid_pixels = 0;
    double depth_sum_m = 0.0;
    float minimum_m = std::numeric_limits<float>::max();
    float maximum_m = 0.0f;
    int captured_frames = 0;

    for (int i = 0; i < 30; ++i)
    {
        auto depth = pipeline.wait_for_frames().get_depth_frame();
        if (!depth)
            continue;

        ++captured_frames;
        for (int y = 0; y < depth.get_height(); ++y)
        {
            for (int x = 0; x < depth.get_width(); ++x)
            {
                const float meters = depth.get_distance(x, y);
                if (meters <= 0.0f)
                    continue;
                ++valid_pixels;
                depth_sum_m += meters;
                minimum_m = std::min(minimum_m, meters);
                maximum_m = std::max(maximum_m, meters);
            }
        }
    }

    pipeline.stop();
    if (captured_frames == 0 || valid_pixels == 0)
        throw std::runtime_error("The camera returned no valid depth samples");

    std::cout << "Depth frames: " << captured_frames << '\n'
              << "Valid samples: " << valid_pixels << '\n'
              << "Depth range: " << minimum_m << " m to " << maximum_m << " m\n"
              << "Mean valid depth: " << (depth_sum_m / valid_pixels) << " m\n";
    return 0;
}
catch (const rs2::error& error)
{
    std::cerr << "RealSense error in " << error.get_failed_function() << ": "
              << error.what() << '\n';
    return 1;
}
catch (const std::exception& error)
{
    std::cerr << "Error: " << error.what() << '\n';
    return 1;
}
