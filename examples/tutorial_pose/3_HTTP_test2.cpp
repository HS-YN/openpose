// ------------------------- OpenPose Library Tutorial - Pose - Example 1 - Extract from Image -------------------------
// This first example shows the user how to:
// 1. Load an image (`filestream` module)
// 2. Extract the pose of that image (`pose` module)
// 3. Render the pose on a resized copy of the input image (`pose` module)
// 4. Display the rendered pose (`gui` module)
// In addition to the previous OpenPose modules, we also need to use:
// 1. `core` module: for the Array<float> class that the `pose` module needs
// 2. `utilities` module: for the error & logging functions, i.e. op::error & op::log respectively

// 3rdparty dependencies
// GFlags: DEFINE_bool, _int32, _int64, _uint64, _double, _string
#include <gflags/gflags.h>
// Allow Google Flags in Ubuntu 14
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif
// OpenPose dependencies
#include <openpose/core/headers.hpp>
#include <openpose/filestream/headers.hpp>
#include <openpose/gui/headers.hpp>
#include <openpose/pose/headers.hpp>
#include <openpose/utilities/headers.hpp>
#include "HTTP/largepost.h"
//#define PORT            8894
#include <iostream>
#include <algorithm>
#include <string>
#include <unistd.h>

//extern "C" {
#include <darknet.h>
//}

#include <google/protobuf/message.h>
#include <google/protobuf/message_lite.h>
#include "bodycoord.pb.h"

// 2018/08/17 In order to use protocol buffer for data communication, it is necessary to
// use specific size of character array, otherwise robot-side cannot decode the buffer
// and create this error:
// "Protocol message tag had invalid wire type"
// Thus, protobuf_results will be allocated(alloc) at Main block,
// then re-allocated everytime the data transaction is made.
char *protobuf_results;

// See all the available parameter options withe the `--help` flag. E.g. `build/examples/openpose/openpose.bin --help`
// Note: This command will show you flags for other unnecessary 3rdparty files. Check only the flags for the OpenPose
// executable. E.g. for `openpose.bin`, look for `Flags from examples/openpose/openpose.cpp:`.
// Debugging/Other
DEFINE_string(darknet_path, "/ROAR/darknet", "Specify the path of darknet");
DEFINE_int32(logging_level, 3, "The logging level. Integer in the range [0, 255]. 0 will output any log() message, while"
                               " 255 will not output any. Current OpenPose library messages are in the range 0-4: 1 for"
                               " low priority messages and 4 for important ones.");
// Producer
DEFINE_string(image_path, "examples/media/COCO_val2014_000000000192.jpg", "Process the desired image.");
// OpenPose
DEFINE_string(model_pose, "COCO", "Model to be used. E.g. `COCO` (18 keypoints), `MPI` (15 keypoints, ~10% faster), "
                                  "`MPI_4_layers` (15 keypoints, even faster but less accurate).");
DEFINE_string(model_folder, "models/", "Folder path (absolute or relative) where the models (pose, face, ...) are located.");
DEFINE_string(net_resolution, "-1x368", "Multiples of 16. If it is increased, the accuracy potentially increases. If it is"
                                        " decreased, the speed increases. For maximum speed-accuracy balance, it should keep the"
                                        " closest aspect ratio possible to the images or videos to be processed. Using `-1` in"
                                        " any of the dimensions, OP will choose the optimal aspect ratio depending on the user's"
                                        " input value. E.g. the default `-1x368` is equivalent to `656x368` in 16:9 resolutions,"
                                        " e.g. full HD (1980x1080) and HD (1280x720) resolutions.");
DEFINE_string(output_resolution, "-1x-1", "The image resolution (display and output). Use \"-1x-1\" to force the program to use the"
                                          " input image resolution.");
DEFINE_int32(num_gpu_start, 0, "GPU device start number.");
DEFINE_double(scale_gap, 0.3, "Scale gap between scales. No effect unless scale_number > 1. Initial scale is always 1."
                              " If you want to change the initial scale, you actually want to multiply the"
                              " `net_resolution` by your desired initial scale.");
DEFINE_int32(scale_number, 1, "Number of scales to average.");
// OpenPose Rendering
DEFINE_bool(disable_blending, false, "If enabled, it will render the results (keypoint skeletons or heatmaps) on a black"
                                     " background, instead of being rendered into the original image. Related: `part_to_show`,"
                                     " `alpha_pose`, and `alpha_pose`.");
DEFINE_double(render_threshold, 0.05, "Only estimated keypoints whose score confidences are higher than this threshold will be"
                                      " rendered. Generally, a high threshold (> 0.5) will only render very clear body parts;"
                                      " while small thresholds (~0.1) will also output guessed and occluded keypoints, but also"
                                      " more false positives (i.e. wrong detections).");
DEFINE_double(alpha_pose, 0.6, "Blending factor (range 0-1) for the body part rendering. 1 will show it completely, 0 will"
                               " hide it. Only valid for GPU rendering.");

DEFINE_bool(ShowRenderedImage, true, "If enabled, it will show the rendered skeleton, but it will inrease the network load.");

DEFINE_uint64(port_number, 8895, "Port number to received HTTP packages" );

op::PoseExtractorCaffe *p_poseExtractorCaffe;
op::PoseCpuRenderer *p_poseRenderer;
bool b_firsttime = true;


/*
 * YOLO-V3 function prototype and global variables for network initialization
 * (2018.08.02 Heeseung Yun)
 */

void yolo_initialize();
void yolo_detect(unsigned char *pFramedata, long JPEG_data_size, bodycoord::Detect *body_coord);

char **yolo_names;
image **yolo_alphabet;
network *yolo_net;

//int openPoseTutorialPose1_modified(unsigned char *pFramedata, char *results)
int openPoseTutorialPose1_modified(unsigned char *pFramedata, long JPEG_data_size, bodycoord::Detect *body_coord)
{
    //op::log("OpenPose Library Tutorial - Example 1.", op::Priority::High);
    // ------------------------- INITIALIZATION -------------------------
    // Step 1 - Set logging level
    // - 0 will output all the logging messages
    // - 255 will output nothing
    op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
              __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // Step 2 - Read Google flags (user defined configuration)
    // outputSize
    const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
    // netInputSize
    const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
    // poseModel
    const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
    // Check no contradictory flags enabled
    if (FLAGS_alpha_pose < 0. || FLAGS_alpha_pose > 1.)
        op::error("Alpha value for blending must be in the range [0,1].", __LINE__, __FUNCTION__, __FILE__);
    if (FLAGS_scale_gap <= 0. && FLAGS_scale_number > 1)
        op::error("Incompatible flag configuration: scale_gap must be greater than 0 or scale_number = 1.",
                  __LINE__, __FUNCTION__, __FILE__);
    // Logging
    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    // Step 3 - Initialize all required classes
    op::ScaleAndSizeExtractor scaleAndSizeExtractor(netInputSize, outputSize, FLAGS_scale_number, FLAGS_scale_gap);
    op::CvMatToOpInput cvMatToOpInput{poseModel};
    op::CvMatToOpOutput cvMatToOpOutput;
    op::OpOutputToCvMat opOutputToCvMat;
    op::FrameDisplayer frameDisplayer{"OpenPose Tutorial - Example 1", outputSize};
    // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
    if (b_firsttime)
    {
        p_poseExtractorCaffe->initializationOnThread();
        p_poseRenderer->initializationOnThread();
        b_firsttime = false;
    }

    std::vector<char> JPEG_Data(pFramedata, pFramedata + JPEG_data_size);
    cv::Mat inputImage = cv::imdecode(JPEG_Data,cv::IMREAD_COLOR);

    //cv::imshow("input image",inputImage);
    //cv::waitKey(1);
    const op::Point<int> imageSize{inputImage.cols, inputImage.rows};
    // Step 2 - Get desired scale sizes
    std::vector<double> scaleInputToNetInputs;
    std::vector<op::Point<int>> netInputSizes;
    double scaleInputToOutput;
    op::Point<int> outputResolution;
    std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution) = scaleAndSizeExtractor.extract(imageSize);
    // Step 3 - Format input image to OpenPose input and output formats
    const auto netInputArray = cvMatToOpInput.createArray(inputImage, scaleInputToNetInputs, netInputSizes);
    auto outputArray = cvMatToOpOutput.createArray(inputImage, scaleInputToOutput, outputResolution);
    // Step 4 - Estimate poseKeypoints
    p_poseExtractorCaffe->forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
    const auto poseKeypoints = p_poseExtractorCaffe->getPoseKeypoints();
    //3/29/2018 The type of poseKeypoints is an Array<float>, it is an Openpose class. See core/Array.hpp
    std::string pose_string = poseKeypoints.toString();
    pose_string.erase(0, 22);
    //replace is for removing "Array<T>::toString():"
    //std::string pose_count = std::to_string(std::count(pose_string.begin(), pose_string.end(), '\n')/19) + "\n";
    //strcpy(results, pose_count.c_str());
    //strcat(results, pose_string.c_str());
    //std::cout << pose_count << pose_string;
    body_coord->set_openpose_cnt(std::count(pose_string.begin(), pose_string.end(), '\n')/19);
    int newline_detect = pose_string.find('\n');
    size_t newline_count = 0, detect_start = 0, detect_length = 0, total_length = pose_string.length();
    while(newline_detect < total_length) {
        newline_count++;
        if(newline_count == 18) {
            detect_length = newline_detect - detect_start;
            body_coord->add_openpose_coord(pose_string.substr(detect_start, detect_length-1));
            detect_start = newline_detect + 2;
            newline_detect++;
            newline_count = 0;
        }
        newline_detect = pose_string.find('\n', newline_detect+1);
    }

    //3/28/2018 I need to pass the Keypoints to Zenbo.
    // Step 5 - Render poseKeypoints
    if (FLAGS_ShowRenderedImage)
    {
        p_poseRenderer->renderPose(outputArray, poseKeypoints, scaleInputToOutput);
        // Step 6 - OpenPose output format to cv::Mat
        auto outputImage = opOutputToCvMat.formatToCvMat(outputArray);

        // ------------------------- SHOWING RESULT AND CLOSING -------------------------
        // Step 1 - Show results 
        // result is disabled now. If you want to see the result from Openpose directly, use the code below:
        //frameDisplayer.displayFrame(outputImage, 1); // Alternative: cv::imshow(outputImage) + cv::waitKey(0)
    }
    // Step 2 - Logging information message
    //op::log("Example 1 successfully finished.", op::Priority::High);

    // Return successful message
    return 0;
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    //Setup protobuf
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    protobuf_results = (char *)malloc(sizeof(char) * 4096);

    //initialize openpose
    // poseModel
    const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
    op::PoseExtractorCaffe poseExtractorCaffe{poseModel, FLAGS_model_folder, FLAGS_num_gpu_start};
    p_poseExtractorCaffe = &poseExtractorCaffe;
    op::PoseCpuRenderer poseRenderer{poseModel, (float)FLAGS_render_threshold, !FLAGS_disable_blending,
                                     (float)FLAGS_alpha_pose};
    p_poseRenderer = &poseRenderer;

    //initialize YOLO
    yolo_initialize();

    struct MHD_Daemon *daemon;

    printf("port_number %d\n", (unsigned short)FLAGS_port_number);
     daemon = MHD_start_daemon(MHD_USE_SELECT_INTERNALLY, (unsigned short)FLAGS_port_number, NULL, NULL,
                              &answer_to_connection, NULL,
                              MHD_OPTION_NOTIFY_COMPLETED, request_completed,
                              NULL, MHD_OPTION_END);
    if (NULL == daemon)
        return 1;
    getchar();
    MHD_stop_daemon(daemon);
    return 0;
}

#include <sys/types.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <microhttpd.h>
#include <time.h>

//void test_detector_http(char *datacfg, network *net, unsigned char *pFramebuffer, float thresh, float hier_thresh, char *outfile, int fullscreen, char *str_results);
static unsigned int nr_of_uploading_clients = 0;

struct connection_info_struct
{
    int connectiontype;
    struct MHD_PostProcessor *postprocessor;
    unsigned char *pFramedata;
    const char *answerstring;
    int answercode;
};

const char *askpage = "<html><body>\n\
                       Upload a file, please!<br>\n\
                       There are %u clients uploading at the moment.<br>\n\
                       <form action=\"/filepost\" method=\"post\" enctype=\"multipart/form-data\">\n\
                       <input name=\"file\" type=\"file\">\n\
                       <input type=\"submit\" value=\" Send \"></form>\n\
                       </body></html>";

const char *busypage =
    "<html><body>This server is busy, please try again later.</body></html>";

const char *completepage =
    "<html><body>The upload has been completed.</body></html>";

const char *errorpage =
    "<html><body>This doesn't seem to be right.</body></html>";
const char *servererrorpage =
    "<html><body>An internal server error has occured.</body></html>";
const char *fileexistspage =
    "<html><body>This file already exists.</body></html>";

const char *in_iteration =
    "<html><body>It is in the iteration.</body></html>";

const char *str_request_complete =
    "<html><body>request completed.</body></html>";

char str_results[4096];

static int
send_page(struct MHD_Connection *connection, const char *page,
          int status_code)
{
    int ret;
    struct MHD_Response *response;

    response =
        MHD_create_response_from_buffer(strlen(page), (void *)page,
                                        MHD_RESPMEM_MUST_COPY);
    if (!response)
        return MHD_NO;
    MHD_add_response_header(response, MHD_HTTP_HEADER_CONTENT_TYPE, "text/html");
    ret = MHD_queue_response(connection, status_code, response);
    MHD_destroy_response(response);

    return ret;
}

static int
iterate_post(void *coninfo_cls, enum MHD_ValueKind kind, const char *key,
             const char *filename, const char *content_type,
             const char *transfer_encoding, const char *data, uint64_t off,
             size_t size)
{
    struct connection_info_struct *con_info = static_cast<connection_info_struct *>(coninfo_cls);
    //  FILE *fp;

    con_info->answerstring = servererrorpage;
    con_info->answercode = MHD_HTTP_INTERNAL_SERVER_ERROR;

    long total_size = atoi(filename);       //The filename field contains the length information
    //The key field containst the timestamp information

    if (size > 0)
    {
        con_info->answerstring = in_iteration;
        memcpy(con_info->pFramedata + off, data, size);
        if (off + size == total_size) //640*480*4
        {
            //Initialize protobuf
            bodycoord::Detect body_coord;
            openPoseTutorialPose1_modified(con_info->pFramedata, total_size, &body_coord);
            yolo_detect(con_info->pFramedata, total_size, &body_coord);
            body_coord.set_key(key);
            //6/1/2018 I copy the piece of code here, to prevent memory leak.
            if (con_info->pFramedata)
            {
                free(con_info->pFramedata); //free memory
                con_info->pFramedata = NULL;
            }
            //std::cout << body_coord.DebugString() << std::endl;
            //if(body_coord.has_yolo_cnt() && body_coord.has_openpose_cnt()) {
            //delete [] con_info->answerstring;
            //std::string protobuf_result;
            //body_coord.SerializeToString(&protobuf_result);
            //con_info->answerstring = protobuf_result.c_str
            //con_info->answerstring = *(char**)protobuf_result;
            
            free(protobuf_results);
            protobuf_results = (char *)malloc(body_coord.ByteSize() * sizeof(char));
            body_coord.MessageLite::SerializeToArray(protobuf_results, body_coord.ByteSize());
            //con_info->answerstring = protobuf_results;
            con_info->answerstring = strcpy(str_results, body_coord.DebugString().c_str());
            printf("%s\n", con_info->answerstring);

            //Debugging
            //body_coord.ParseFromString(con_info->answerstring);
            //std::cout << body_coord.DebugString() << std::endl;
            
            //}
            //else {
            //    con_info->answerstring = strcpy(str_results, "");
            //}
            //con_info->answerstring = strcpy(str_results, key);
        }
    }

    con_info->answercode = MHD_HTTP_OK;

    return MHD_YES;
}

//3/9/2018 Chih-Yuan : this is a callback function. It is too late the change the answerstring here.
void request_completed(void *cls, struct MHD_Connection *connection,
                       void **con_cls, enum MHD_RequestTerminationCode toe)
{
    struct connection_info_struct *con_info = static_cast<connection_info_struct *>(*con_cls);

    if (NULL == con_info)
        return;

    if (con_info->connectiontype == POST)
    {
        if (NULL != con_info->postprocessor)
        {
            MHD_destroy_post_processor(con_info->postprocessor);
            nr_of_uploading_clients--;
        }

        //Here should be the place I call TinyYOLO.
        //	  time_t rawtime;
        //	  struct tm * timeinfo;
        //	  time( &rawtime);
        //	  timeinfo = localtime( &rawtime);
        //	  printf("Current local time and date: %s", asctime(timeinfo));
        //		printf("A frame has been uploaded\n");

        if (con_info->pFramedata)
        {
            free(con_info->pFramedata); //free memory
            con_info->pFramedata = NULL;
        }
    }

    if (con_info->pFramedata)
    {
        free(con_info->pFramedata); //free memory
        con_info->pFramedata = NULL;
    }

    free(con_info);
    *con_cls = NULL;
}

//4/20/2018 Chih-Yuan: This is the first function used when a new POST request is coming.
int answer_to_connection(void *cls, struct MHD_Connection *connection,
                         const char *url, const char *method,
                         const char *version, const char *upload_data,
                         size_t *upload_data_size, void **con_cls)
{

    if (NULL == *con_cls)
    {
        struct connection_info_struct *con_info;

        if (nr_of_uploading_clients >= MAXCLIENTS)
            return send_page(connection, busypage, MHD_HTTP_SERVICE_UNAVAILABLE);

        con_info = static_cast<connection_info_struct *>(malloc(sizeof(struct connection_info_struct)));
        if (NULL == con_info)
            return MHD_NO;

        //con_info->fp = NULL;
        con_info->pFramedata = NULL; //initialize

        if (0 == strcmp(method, "POST"))
        {
            con_info->postprocessor =
                MHD_create_post_processor(connection, POSTBUFFERSIZE,
                                          iterate_post, (void *)con_info);

            if (NULL == con_info->postprocessor)
            {
                free(con_info);
                return MHD_NO;
            }

            nr_of_uploading_clients++;

            con_info->connectiontype = POST;
            con_info->answercode = MHD_HTTP_OK;
            con_info->answerstring = completepage;
            //allocate buffer here
            size_t datasize = 640 * 480 * 4;//4/20/2018 Chih-Yuan: I don't need to change it right now.
            con_info->pFramedata = static_cast<unsigned char *>(malloc(datasize));      //6/1/2018 check it. Whether it causes memory leak.
        }
        else
            con_info->connectiontype = GET;

        *con_cls = (void *)con_info;

        return MHD_YES;
    }

    if (0 == strcmp(method, "GET"))
    {
        char buffer[1024];

        snprintf(buffer, sizeof(buffer), askpage, nr_of_uploading_clients);
        return send_page(connection, buffer, MHD_HTTP_OK);
    }

    if (0 == strcmp(method, "POST"))
    {
        struct connection_info_struct *con_info = static_cast<connection_info_struct *>(*con_cls);

        if (0 != *upload_data_size)
        {
            MHD_post_process(con_info->postprocessor, upload_data,
                             *upload_data_size);
            *upload_data_size = 0;

            return MHD_YES;
        }
        else
        {
            //if (NULL != con_info->fp)
            //  fclose (con_info->fp);		//what is the meaning of this statements? POST but upload_data_size is 0?

            /* Now it is safe to open and inspect the file before calling send_page with a response */
            return send_page(connection, con_info->answerstring,
                             con_info->answercode);
        }
    }

    return send_page(connection, errorpage, MHD_HTTP_BAD_REQUEST);
}

void yolo_initialize() {
    // Initialize environment variable that need not change throughout the execution
    char *datacfg = "cfg/coco.data";
    char *cfg = "cfg/yolov3.cfg";
    char *weights = "yolov3.weights";
    char *curr_dir = get_current_dir_name();

    // Change current directory to handle darknet-dependent functions
    chdir(FLAGS_darknet_path.c_str());

    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    yolo_names = get_labels(name_list);

    yolo_alphabet = load_alphabet();
    yolo_net = load_network(cfg, weights, 0); 
    set_batch_network(yolo_net, 1); 
    srand(2222222);

    chdir(curr_dir);
}

void yolo_detect(unsigned char *pFramedata, long JPEG_data_size, bodycoord::Detect *body_coord) {
    // yolo_names, yolo_alphabet, yolo_net as global variables
    const float thresh = 0.5, nms = 0.45;
    double time = what_time_is_it_now();
    // Convert CV::Mat to darknet image
    // Refer to darknet/src/image.c
    
    // (1) Initialize Iplimage ( 01 ms)
    
    // Using IplImage is fast! more than three times faster.
    // But there are severe caveats:
    // First is (random) Segmentation Fault without any explicit reason
    // (Maybe due to inner data formatting problem)
    // Second is Image Corruption
    // Therefore, for initialization, it would be safe to use cv::Mat 
    // rather than obsolete IplImage in terms of robustness.
    // IplImage* img_temp = new IplImage(cv::imdecode(JPEG_Data, cv::IMREAD_COLOR));
    // IplImage* resized = cvCreateImage(cvSize(416, 311), img_temp->depth, img_temp->nChannels);
    // cvResize(img_temp, resized);

    std::vector<char> JPEG_Data(pFramedata, pFramedata + JPEG_data_size);
    cv::Mat img_raw = cv::imdecode(JPEG_Data, cv::IMREAD_COLOR);
    cv::Mat img_downscale;
    cv::resize(img_raw, img_downscale, cv::Size(416, 312), 0, 0, cv::INTER_CUBIC);

    IplImage resized_orig = img_downscale;
    IplImage *resized = &resized_orig;
    //416x312 was set in accordance with 640x480, original zenbo camera resolution
    //for better code, it is better not to use magic number like below
    //note that computation is way more efficient for uchar than float
    //thus, letterbox_image(im, yolo_net->w, yolo_net->h) is replaced with below

    unsigned char *data = (unsigned char *)resized->imageData;
    int h = resized->height, w = resized->width, c = resized->nChannels;
    int step = resized->widthStep;
    
    // (2) Converting image ( 03 ms)
    image im = make_image(yolo_net->w, yolo_net->h, c); 
    // Note that darknet color channel is like BGR, not RGB.
    // instead of [c*w*h + i*w + j] as given in the original darknet code,
    // modify c into 'c-1-k' so that the channel mapping is inverted.
    for(int i = 0; i < h; ++i)
        for(int k = 0; k < c; ++k)
            for(int j = 0; j < w; ++j)
                im.data[k*w*w + (i + 52)*w + j] = data[i*step + j*c + (c-1-k)]/255.;
    
    for(int i = 0; i < 52; ++i) {
        for(int k = 0; k < c; ++k) {
            for(int j = 0; j < w; ++j) {
                im.data[k*w*w + i*w + j] = 0.5;
                im.data[k*w*w + (i + 364)*w + j] = 0.5;
            }
        }
    }
 
    // (3) Resizing image ( 30 ms)
    //image sized = letterbox_image(im, yolo_net->w, yolo_net->h);
    //after optimization, the aggregated classification time was reduced from 54ms to 22ms
    image sized = im;
    layer l = yolo_net->layers[yolo_net->n-1];

    save_image(im, "pred");
    
    // (4) Prediction ( 17 ms)
    float *X = sized.data;
    float* out = network_predict(yolo_net, X); 
    int nboxes = 0;
    // (5) Counting & Drawing Bboxes ( 03 ms)
    detection *dets = get_network_boxes(yolo_net, im.w, im.h, thresh, thresh, 0, 1, &nboxes);
    if(nms)
        do_nms_sort(dets, nboxes, l.classes, nms);
    //draw_detections(im, dets, nboxes, thresh, yolo_names, yolo_alphabet, l.classes);

    // from image.c:239 draw_detections
    // names[0], dets[i].prob[0] indicates person
    char person_point[1024];
    int person_cnt = 0;
    for(int i = 0; i < nboxes; ++i) {
        if(dets[i].prob[0] > thresh) {
            //printf("person detected");
            box b = dets[i].bbox;
            //printf("\t x:%f y:%f w:%f h:%f\n", b.x, b.y, b.w, b.h);
            sprintf(person_point, "%f, %f, %f, %f\n", b.x, b.y, b.w, b.h);
            body_coord->add_yolo_coord(person_point);
            person_cnt++;
            //strcat(results, person_point);
        }
    }
    body_coord->set_yolo_cnt(person_cnt);
    
    free_detections(dets, nboxes);

    //save_image(im, "pred");

    //free_image(sized);
    free_image(im);
    //cvReleaseImage(&img_temp);

    //printf("Predicted in %f seconds. \n", what_time_is_it_now() - time);
}

/*
 * Due to protobuf incompatibility (3.3.0-python versus 2.6.1-c++)
 * this portion of code cannot be used.
 *
void charades_webcam() {
    int argc = 1;
    char *argv[] = {"./build/examples/tutorial_pose/3_HTTP_test2.bin"};
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    FILE *fd;

    wchar_t **wargv;
    wargv = (wchar_t**)malloc(1 * sizeof(wchar_t*));
    *wargv = (wchar_t*)malloc(6 * sizeof(wchar_t));
    **wargv = L'argv1';

    if(program == NULL)
        perror("Error from decoding locale\n");

    fd = fopen("/Net/charades-webcam/charades_webcam.py", "r");
    if(fd == NULL)
        perror("Error loading charades-webcam.py, aborting...\n");
    
    Py_SetProgramName(program);
    Py_Initialize();
    
    //This line is causing trouble...
    PySys_SetArgv(1, (wchar_t**)wargv);

    PyRun_SimpleString("import sys\n");
    PyRun_SimpleString("sys.path.append('/opt/conda/envs/object-detection/lib/python3.5/site-packages/')\n");
    PyRun_SimpleString("sys.path.append('$HOME/charades-webcam/')\n");

    PyRun_SimpleFile(fd, "/Net/charades-webcam/charades_webcam.py");

    std::cout << "FIIIIIINISHED!!!!!" << std::endl;

    Py_Finalize();
    PyMem_RawFree(program);
}
*/
