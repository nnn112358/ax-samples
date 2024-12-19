/*
* AXERA is pleased to support the open source community by making ax-samples available.
*
* Copyright (c) 2022, AXERA Semiconductor (Shanghai) Co., Ltd. All rights reserved.
*
* Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
* in compliance with the License. You may obtain a copy of the License at
*
* https://opensource.org/licenses/BSD-3-Clause
*
* Unless required by applicable law or agreed to in writing, software distributed
* under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
* CONDITIONS OF ANY KIND, either express or implied. See the License for the
* specific language governing permissions and limitations under the License.
*/

/*
* Author: ZHEQIUSHUI
*/

/*
* added: @nnn112358
*/


#include <cstdio>
#include <cstring>
#include <numeric>
#include <chrono>
#include <thread>

#include <opencv2/opencv.hpp>
#include "base/common.hpp"
#include "middleware/io.hpp"

#include "utilities/args.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include "utilities/timer.hpp"

#include <ax_sys_api.h>
#include <ax_engine_api.h>

#include "base/score.hpp"
#include "base/topk.hpp"

const int DEFAULT_IMG_H = 224;
const int DEFAULT_IMG_W = 224;
const int DEFAULT_LOOP_COUNT = 1;

#include <mjpeg_streamer.hpp>
using MJPEGStreamer = nadjieb::MJPEGStreamer;

// ミリ秒単位でスリープ
void msleep(unsigned int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

namespace ax
{
    void post_process(AX_ENGINE_IO_INFO_T* io_info, AX_ENGINE_IO_T* io_data, const cv::Mat& mat, const float time_cost)
    {
        timer timer_postprocess;

        auto& output = io_data->pOutputs[0];
        auto& info = io_info->pOutputs[0];
        auto ptr = (float*)output.pVirAddr;
        auto class_num = info.nSize / sizeof(float);
        std::vector<classification::score> result(class_num);
        for (uint32_t id = 0; id < class_num; id++)
        {
            result[id].id = id;
            result[id].score = ptr[id];
        }
        classification::sort_score(result);
        fprintf(stdout, "time_costs:%.2f ms \n", time_cost);
        fprintf(stdout, "post_process time:%.2f ms \n", timer_postprocess.cost());
        classification::print_score(result, 5);
        fprintf(stdout, "--------------------------------------\n");

    }

    bool run_model_on_video(const std::string& model, const std::string& video_path, const std::array<int, 2>& input_size)
    {
		//動画ファイルorカメラデバイスをOpen
		cv::VideoCapture cap;

	    // 文字列が数字だけで構成されているか確認
	    bool isNumber = !video_path.empty() && 
	                   std::all_of(video_path.begin(), video_path.end(), ::isdigit);

	    if (isNumber) {
	        // 数字の場合はカメラデバイスIDとして解釈
	        int deviceID = std::stoi(video_path);
			cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
			cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	        cap.open(deviceID);
	        if (!cap.isOpened()) {
	            std::cerr << "Camera " << deviceID << " Not Open" << std::endl;
	        }
	    } else {
	        // 数字以外が含まれる場合はファイルパスとして解釈
	        //cap.open(video_path);
	        //if (!cap.isOpened()) {
	        //    std::cerr << "Video " << video_path << " Not Open" << std::endl;
	        //}
			return -1;
	    }

        // 1. init engine
        AX_ENGINE_NPU_ATTR_T npu_attr;
        memset(&npu_attr, 0, sizeof(npu_attr));
        npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        auto ret = AX_ENGINE_Init(&npu_attr);
        if (0 != ret) return ret;

        // 2. load model
        std::vector<char> model_buffer;
        if (!utilities::read_file(model, model_buffer)) {
            fprintf(stderr, "Read Run-Joint model(%s) file failed.\n", model.c_str());
            return false;
        }

        // 3. create handle
        AX_ENGINE_HANDLE handle;
        ret = AX_ENGINE_CreateHandle(&handle, model_buffer.data(), model_buffer.size());
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine creating handle is done.\n");

        // 4. create context
        ret = AX_ENGINE_CreateContext(handle);
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine creating context is done.\n");

        // 5. set io
        AX_ENGINE_IO_INFO_T* io_info;
        ret = AX_ENGINE_GetIOInfo(handle, &io_info);
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine get io info is done. \n");

        // 6. alloc io
        AX_ENGINE_IO_T io_data;
        ret = middleware::prepare_io(io_info, &io_data, std::make_pair(AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED));
        SAMPLE_AX_ENGINE_DEAL_HANDLE
        fprintf(stdout, "Engine alloc io is done. \n");

        // Process video frames
        cv::Mat frame;
        std::vector<uint8_t> image_data(input_size[0] * input_size[1] * 3, 0);

		//MJPEGStreamer Initialize
		MJPEGStreamer streamer;
		std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 70};
		streamer.start(7777);

        while (cap.isOpened()) {

			cv::Mat frame;
			cap >> frame;
			cv::resize(frame, frame, cv::Size(320, 240)); // 320x240にリサイズ

            // Prepare input data
            common::get_input_data_centercrop(frame, image_data, input_size[0], input_size[1]);
            
            // Push input and run inference
            ret = middleware::push_input(image_data, &io_data, io_info);
            SAMPLE_AX_ENGINE_DEAL_HANDLE_IO
            
            timer tick;
            ret = AX_ENGINE_RunSync(handle, &io_data);
            float time_cost = tick.cost();
            
            // Post process single frame
            post_process(io_info, &io_data, frame, time_cost);

			std::vector<uchar> buff_bgr;
			cv::imencode(".jpg", frame, buff_bgr, params);
			streamer.publish("/video", std::string(buff_bgr.begin(), buff_bgr.end()));

			msleep(10);  

        }
	    streamer.stop();

        // Cleanup
        cap.release();
        middleware::free_io(&io_data);
        AX_ENGINE_DestroyHandle(handle);
        return true;
    }
} // namespace ax

int main(int argc, char* argv[])
{
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "joint file(a.k.a. joint model)", true, "");
    cmd.add<std::string>("video", 'v', "video file", false, "0");
    cmd.add<std::string>("size", 'g', "input_h, input_w", false, std::to_string(DEFAULT_IMG_H) + "," + std::to_string(DEFAULT_IMG_W));
    cmd.parse_check(argc, argv);

    auto model_file = cmd.get<std::string>("model");
    auto video_file = cmd.get<std::string>("video");

    auto input_size_string = cmd.get<std::string>("size");
    std::array<int, 2> input_size = {DEFAULT_IMG_H, DEFAULT_IMG_W};
    if (!utilities::parse_string(input_size_string, input_size)) {
        fprintf(stderr, "Invalid input size format\n");
        return -1;
    }

    // 1. print args
    fprintf(stdout, "--------------------------------------\n");
    fprintf(stdout, "Model file: %s\n", model_file.c_str());
    fprintf(stdout, "Video file: %s\n", video_file.c_str());
    fprintf(stdout, "Input size (h,w): %d,%d\n", input_size[0], input_size[1]);
    fprintf(stdout, "--------------------------------------\n");

    // 2. sys_init
    AX_SYS_Init();

    // 3. -  engine model  -  can only use AX_ENGINE** inside
    {
        // AX_ENGINE_NPUReset(); // todo ??
    	ax::run_model_on_video(model_file, video_file, input_size);

        // 4.3 engine de init
        AX_ENGINE_Deinit();
        // AX_ENGINE_NPUReset();
    }
    // 4. -  engine model  -

    AX_SYS_Deinit();
    return 0;
}
