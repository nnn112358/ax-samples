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

#include <base/mjpeg_streamer.hpp>
using MJPEGStreamer = nadjieb::MJPEGStreamer;

const int DEFAULT_IMG_H = 256;
const int DEFAULT_IMG_W = 384;

// ミリ秒単位でスリープ
void msleep(unsigned int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

namespace ax
{
      bool process_video_frame(AX_ENGINE_HANDLE handle, AX_ENGINE_IO_INFO_T* io_info, AX_ENGINE_IO_T* io_data, 
                            cv::Mat& frame, cv::Mat& out_img, int input_h, int input_w) {

        std::vector<uint8_t> image(input_h * input_w * 3, 0);
        common::get_input_data_letterbox(frame, image, input_h, input_w,true);
        
        auto ret = middleware::push_input(image, io_data, io_info);
        if (ret != 0) return false;

        timer tick;
        ret = AX_ENGINE_RunSync(handle, io_data);

        timer timer_postprocess;
        auto& output = io_data->pOutputs[0];
        auto& info = io_info->pOutputs[0];

        cv::Mat feature(info.pShape[2], info.pShape[3], CV_32FC1, output.pVirAddr);

        double minVal, maxVal;
        cv::minMaxLoc(feature, &minVal, &maxVal);

		static double minVal_ave=0;
		static double maxVal_ave=0;

		if(minVal_ave==0)minVal_ave=minVal;
		if(maxVal_ave==0)maxVal_ave=maxVal;

		minVal_ave=0.5*minVal_ave+0.5*minVal;
		maxVal_ave=0.5*maxVal_ave+0.5*maxVal;

		feature -= minVal_ave;
		feature /= (maxVal_ave - minVal_ave);
		// feature = 1.f - feature;
		feature *= 255;

        feature.convertTo(feature, CV_8UC1);

        cv::Mat dst(info.pShape[2], info.pShape[3], CV_8UC3);
        cv::applyColorMap(feature, dst, cv::ColormapTypes::COLORMAP_MAGMA);
        cv::resize(dst, dst, cv::Size(frame.cols, frame.rows));

        cv::hconcat(std::vector<cv::Mat>{frame, dst}, dst);
        out_img=dst.clone();
        return true;
    }
    

    bool run_model(const std::string& model, const std::string& video_path, const std::array<int, 2>& input_size) {

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

		std::chrono::steady_clock::time_point Cbegin, Cend;
		std::chrono::steady_clock::time_point Tbegin, Tend;

	        while (true) {
				Cbegin = std::chrono::steady_clock::now();

				cv::Mat frame,out_img;
				cap >> frame;
				if (frame.empty()) break;


				Cend = std::chrono::steady_clock::now();
				Tbegin = std::chrono::steady_clock::now();
				if (!process_video_frame(handle, io_info, &io_data, frame,out_img, input_size[0] ,  input_size[1])) {
				     break;
				 }
	         
	         
			Tend = std::chrono::steady_clock::now();
			float cap_fps = std::chrono::duration_cast<std::chrono::milliseconds>(Cend - Cbegin).count();
			float t_fps = std::chrono::duration_cast<std::chrono::milliseconds>(Tend - Tbegin).count();

	        putText(out_img, cv::format("Camera %0.2f", cap_fps), cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255));
	        putText(out_img, cv::format("Proc %0.2f", t_fps), cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 255));

			std::vector<uchar> buff_bgr1;
			cv::imencode(".jpg", frame, buff_bgr1, params);
			streamer.publish("/frame", std::string(buff_bgr1.begin(), buff_bgr1.end()));

			std::vector<uchar> buff_bgr2;
			cv::imencode(".jpg", out_img, buff_bgr2, params);
			streamer.publish("/video", std::string(buff_bgr2.begin(), buff_bgr2.end()));

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
    cmd.add<std::string>("video", 'v', "video file or camera index (0 for default camera)", true, "");
    cmd.add<std::string>("size", 'g', "input_h, input_w", false, std::to_string(DEFAULT_IMG_H) + "," + std::to_string(DEFAULT_IMG_W));

    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto model_file = cmd.get<std::string>("model");
    auto video_source = cmd.get<std::string>("video");

    auto model_file_flag = utilities::file_exist(model_file);
 //   auto image_file_flag = utilities::file_exist(image_file);

    if (!utilities::file_exist(model_file)) {
        fprintf(stderr, "Model file not found: %s\n", model_file.c_str());
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
    fprintf(stdout, "model file : %s\n", model_file.c_str());
    fprintf(stdout, "img_h, img_w : %d %d\n", input_size[0], input_size[1]);
    fprintf(stdout, "--------------------------------------\n");

    // 2. read image & resize & transpose
    //std::vector<uint8_t> image(input_size[0] * input_size[1] * 3, 0);
    //common::get_input_data_no_letterbox(mat, image, input_size[0], input_size[1], true);

    // 3. sys_init
    AX_SYS_Init();

    // 4. -  engine model  -  can only use AX_ENGINE** inside
    {
        // AX_ENGINE_NPUReset(); // todo ??
        ax::run_model(model_file,  video_source, input_size);

        // 4.3 engine de init
        AX_ENGINE_Deinit();
        // AX_ENGINE_NPUReset();
    }
    // 4. -  engine model  -

    AX_SYS_Deinit();
    return 0;
}
