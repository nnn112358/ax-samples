/*
* AXERA is pleased to support the open source community by making ax-samples available.
*
* Copyright (c) 2024, AXERA Semiconductor Co., Ltd. All rights reserved.
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
* Note: For the YOLO11 series exported by the ultralytics project.
* Author: QQC
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
#include "base/detection.hpp"
#include "middleware/io.hpp"

#include "utilities/args.hpp"
#include "utilities/cmdline.hpp"
#include "utilities/file.hpp"
#include "utilities/timer.hpp"

#include <ax_sys_api.h>
#include <ax_engine_api.h>

const int DEFAULT_IMG_H = 640;
const int DEFAULT_IMG_W = 640;

#include <mjpeg_streamer.hpp>
using MJPEGStreamer = nadjieb::MJPEGStreamer;


// ミリ秒単位でスリープ
void msleep(unsigned int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}


const char* CLASS_NAMES[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"};

int NUM_CLASS = 80;

const int DEFAULT_LOOP_COUNT = 1;

const float PROB_THRESHOLD = 0.45f;
const float NMS_THRESHOLD = 0.45f;
namespace ax
{
    void post_process(AX_ENGINE_IO_INFO_T* io_info, AX_ENGINE_IO_T* io_data, const cv::Mat& mat, cv::Mat& out_img, 
int input_w, int input_h, const float time_cost)
    {
        std::vector<detection::Object> proposals;
        std::vector<detection::Object> objects;
        timer timer_postprocess;
        for (int i = 0; i < 3; ++i)
        {
            auto feat_ptr = (float*)io_data->pOutputs[i].pVirAddr;
            int32_t stride = (1 << i) * 8;
            detection::generate_proposals_yolov8_native(stride, feat_ptr, PROB_THRESHOLD, proposals, input_w, input_h, NUM_CLASS);
        }

        detection::get_out_bbox(proposals, objects, NMS_THRESHOLD, input_h, input_w, mat.rows, mat.cols);
        fprintf(stdout, "time_costs:%.2f ms \n", time_cost);
        fprintf(stdout, "post process cost time:%.2f ms \n", timer_postprocess.cost());
        fprintf(stdout, "--------------------------------------\n");
        fprintf(stdout, "detection num: %zu\n", objects.size());

        detection::draw_objects(mat, out_img, objects, CLASS_NAMES, "yolo11_out");
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
        if (0 != ret)
        {
            return ret;
        }

        // 2. load model
        std::vector<char> model_buffer;
        if (!utilities::read_file(model, model_buffer))
        {
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

			cv::Mat frame,out_img;
			cap >> frame;
			//cv::resize(frame, frame, cv::Size(320, 240)); // 320x240にリサイズ

            // Prepare input data
    		std::vector<uint8_t> image(input_size[0] * input_size[1] * 3, 0);
		    common::get_input_data_letterbox(frame, image, input_size[0], input_size[1]);

            // Push input and run inference
            ret = middleware::push_input(image, &io_data, io_info);
            SAMPLE_AX_ENGINE_DEAL_HANDLE_IO
            
            timer tick;
            ret = AX_ENGINE_RunSync(handle, &io_data);
            float time_cost = tick.cost();
            
            // Post process single frame
        	post_process(io_info, &io_data, frame, out_img,input_size[1], input_size[0], time_cost);

			std::vector<uchar> buff_bgr;
			cv::imencode(".jpg", out_img, buff_bgr, params);
			streamer.publish("/video", std::string(buff_bgr.begin(), buff_bgr.end()));

			msleep(10);  

        }
	    streamer.stop();


        middleware::free_io(&io_data);
        return AX_ENGINE_DestroyHandle(handle);
    }
} // namespace ax

int main(int argc, char* argv[])
{
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "joint file(a.k.a. joint model)", true, "");
    cmd.add<std::string>("video", 'v', "video file", false, "0");
    cmd.add<std::string>("size", 'g', "input_h, input_w", false, std::to_string(DEFAULT_IMG_H) + "," + std::to_string(DEFAULT_IMG_W));
    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto model_file = cmd.get<std::string>("model");
    auto video_file = cmd.get<std::string>("video");

    auto model_file_flag = utilities::file_exist(model_file);

     if (!model_file_flag) {
		 	fprintf(stderr,"model file none");
			return -1;
	}

    auto input_size_string = cmd.get<std::string>("size");
    std::array<int, 2> input_size = {DEFAULT_IMG_H, DEFAULT_IMG_W};
    auto input_size_flag = utilities::parse_string(input_size_string, input_size);

    if (!input_size_flag)
    {
        auto show_error = [](const std::string& kind, const std::string& value) {
            fprintf(stderr, "Input %s(%s) is not allowed, please check it.\n", kind.c_str(), value.c_str());
        };

        show_error("size", input_size_string);

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

        // 3.3 engine de init
        AX_ENGINE_Deinit();
        // AX_ENGINE_NPUReset();
    }
    // 4. -  engine model  -

    AX_SYS_Deinit();
    return 0;
}
