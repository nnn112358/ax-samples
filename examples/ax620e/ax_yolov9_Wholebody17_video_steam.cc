/*
* Author: @nnn112358
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

#include <mjpeg_streamer.hpp>
using MJPEGStreamer = nadjieb::MJPEGStreamer;

const int DEFAULT_IMG_H = 480;
const int DEFAULT_IMG_W = 640;

const char* CLASS_NAMES[] = {
"body","adult","child","male","female","body_with_wheelchair","body_with_crutches","head","face","eye",
"nose","mouth","ear","hand","hand_left","hand_right","foot"};

int NUM_CLASS = 17;


const float PROB_THRESHOLD = 0.40f;
const float NMS_THRESHOLD = 0.25f;

// ミリ秒単位でスリープ
void msleep(unsigned int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

namespace ax
{

  bool process_video_frame(AX_ENGINE_HANDLE handle, AX_ENGINE_IO_INFO_T* io_info, AX_ENGINE_IO_T* io_data, 
                           const cv::Mat& frame, cv::Mat& out_img, int input_h, int input_w) {
        std::vector<uint8_t> image(input_h * input_w * 3, 0);
        common::get_input_data_letterbox(frame, image, input_h, input_w);
        
        auto ret = middleware::push_input(image, io_data, io_info);
        if (ret != 0) return false;
        
        timer tick;
        ret = AX_ENGINE_RunSync(handle, io_data);
        float time_cost = tick.cost();
        
        std::vector<detection::Object> proposals;
        std::vector<detection::Object> objects;
        
        for (int i = 0; i < 3; ++i) {
            auto feat_ptr = (float*)io_data->pOutputs[i].pVirAddr;
            int32_t stride = (1 << i) * 8;
            detection::generate_proposals_yolov9(stride, feat_ptr, PROB_THRESHOLD, proposals, input_w, input_h, NUM_CLASS);
        }
        
        detection::get_out_bbox(proposals, objects, NMS_THRESHOLD, input_h, input_w, frame.rows, frame.cols);
        detection::draw_objects(frame,out_img, objects, CLASS_NAMES, "");
        
        return true;
    }

    bool run_model(const std::string& model, const std::string& video_path,const std::array<int, 2>& input_size) {

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
		streamer.start(7778);

	    std::chrono::steady_clock::time_point Cbegin, Cend;
	    std::chrono::steady_clock::time_point Tbegin, Tend;


        while (cap.isOpened()) {
			Cbegin = std::chrono::steady_clock::now();

			cv::Mat frame,out_img;
			cap >> frame;
			if (frame.empty()) break;

			Cend = std::chrono::steady_clock::now();

			Tbegin = std::chrono::steady_clock::now();
			if (!process_video_frame(handle, io_info, &io_data, frame,out_img,input_size[1], input_size[0])) {
			     break;
			 }
			Tend = std::chrono::steady_clock::now();

		    float cap_fps = std::chrono::duration_cast<std::chrono::milliseconds>(Cend - Cbegin).count();
		    float t_fps = std::chrono::duration_cast<std::chrono::milliseconds>(Tend - Tbegin).count();

	        putText(out_img, cv::format("Camera %0.2f", cap_fps), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255));
	        putText(out_img, cv::format("Process %0.2f", t_fps), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255));

			std::vector<uchar> buff_bgr;
			cv::imencode(".jpg", out_img, buff_bgr, params);
			streamer.publish("/video", std::string(buff_bgr.begin(), buff_bgr.end()));


        }
	    streamer.stop();
        cap.release();
        
        middleware::free_io(&io_data);
        return AX_ENGINE_DestroyHandle(handle);
    }
} // namespace ax
int main(int argc, char* argv[]) {
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "joint file(a.k.a. joint model)", true, "");
    cmd.add<std::string>("video", 'v', "video file or camera index (0 for default camera)", true, "");
    cmd.add<std::string>("size", 'g', "input_h, input_w", false, 
                        std::to_string(DEFAULT_IMG_H) + "," + std::to_string(DEFAULT_IMG_W));

    cmd.parse_check(argc, argv);

    auto model_file = cmd.get<std::string>("model");
    auto video_source = cmd.get<std::string>("video");
    
    if (!utilities::file_exist(model_file)) {
        fprintf(stderr, "Model file not found: %s\n", model_file.c_str());
        return -1;
    }

    std::array<int, 2> input_size = {DEFAULT_IMG_H, DEFAULT_IMG_W};
    auto input_size_string = cmd.get<std::string>("size");
    auto input_size_flag = utilities::parse_string(input_size_string, input_size);

    if (!input_size_flag) {
        fprintf(stderr, "Invalid input size: %s\n", input_size_string.c_str());
        return -1;
    }

    AX_SYS_Init();
    
    ax::run_model(model_file, video_source, input_size);
    
    AX_ENGINE_Deinit();
    AX_SYS_Deinit();
    
    return 0;
}
