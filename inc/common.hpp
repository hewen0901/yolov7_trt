#ifndef ENOCR_CPP_COMMON_COMMON_H_
#define ENOCR_CPP_COMMON_COMMON_H_

#include <cuda_runtime_api.h>
#include <numeric>
#include <fstream>
#include <dirent.h>
#include <NvInferRuntimeCommon.h>
#include "NvOnnxParser.h"
#include "inc/logging.h"

//printf("\033[32m%s (%d) - <%s>\033[0m\n", str);
#define LOG() (printf("\033[32m%s (%d) - <%s>\033[0m\n",__FILE__,__LINE__,__FUNCTION__))

namespace siran
{

static const float color_list[80][3] =
{
    {0.000, 0.447, 0.741},
    {0.850, 0.325, 0.098},
    {0.929, 0.694, 0.125},
    {0.494, 0.184, 0.556},
    {0.466, 0.674, 0.188},
    {0.301, 0.745, 0.933},
    {0.635, 0.078, 0.184},
    {0.300, 0.300, 0.300},
    {0.600, 0.600, 0.600},
    {1.000, 0.000, 0.000},
    {1.000, 0.500, 0.000},
    {0.749, 0.749, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.333, 0.333, 0.000},
    {0.333, 0.667, 0.000},
    {0.333, 1.000, 0.000},
    {0.667, 0.333, 0.000},
    {0.667, 0.667, 0.000},
    {0.667, 1.000, 0.000},
    {1.000, 0.333, 0.000},
    {1.000, 0.667, 0.000},
    {1.000, 1.000, 0.000},
    {0.000, 0.333, 0.500},
    {0.000, 0.667, 0.500},
    {0.000, 1.000, 0.500},
    {0.333, 0.000, 0.500},
    {0.333, 0.333, 0.500},
    {0.333, 0.667, 0.500},
    {0.333, 1.000, 0.500},
    {0.667, 0.000, 0.500},
    {0.667, 0.333, 0.500},
    {0.667, 0.667, 0.500},
    {0.667, 1.000, 0.500},
    {1.000, 0.000, 0.500},
    {1.000, 0.333, 0.500},
    {1.000, 0.667, 0.500},
    {1.000, 1.000, 0.500},
    {0.000, 0.333, 1.000},
    {0.000, 0.667, 1.000},
    {0.000, 1.000, 1.000},
    {0.333, 0.000, 1.000},
    {0.333, 0.333, 1.000},
    {0.333, 0.667, 1.000},
    {0.333, 1.000, 1.000},
    {0.667, 0.000, 1.000},
    {0.667, 0.333, 1.000},
    {0.667, 0.667, 1.000},
    {0.667, 1.000, 1.000},
    {1.000, 0.000, 1.000},
    {1.000, 0.333, 1.000},
    {1.000, 0.667, 1.000},
    {0.333, 0.000, 0.000},
    {0.500, 0.000, 0.000},
    {0.667, 0.000, 0.000},
    {0.833, 0.000, 0.000},
    {1.000, 0.000, 0.000},
    {0.000, 0.167, 0.000},
    {0.000, 0.333, 0.000},
    {0.000, 0.500, 0.000},
    {0.000, 0.667, 0.000},
    {0.000, 0.833, 0.000},
    {0.000, 1.000, 0.000},
    {0.000, 0.000, 0.167},
    {0.000, 0.000, 0.333},
    {0.000, 0.000, 0.500},
    {0.000, 0.000, 0.667},
    {0.000, 0.000, 0.833},
    {0.000, 0.000, 1.000},
    {0.000, 0.000, 0.000},
    {0.143, 0.143, 0.143},
    {0.286, 0.286, 0.286},
    {0.429, 0.429, 0.429},
    {0.571, 0.571, 0.571},
    {0.714, 0.714, 0.714},
    {0.857, 0.857, 0.857},
    {0.000, 0.447, 0.741},
    {0.314, 0.717, 0.741},
    {0.50, 0.5, 0}
};

static const char* class_names[] =
{
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};


/**
 * @brief GaofaSetCudaDevice -- Select algorithm running platform.
 * @param deviceID           -- GPU ID, e.g. support input 0 or 1
 */
static void SetCudaDevice(const int &deviceID)
{
    int countGpus=0;
    cudaGetDeviceCount(&countGpus);
    //std::cout<<"########## gpu num "<<countGpus<<" ##########"<<std::endl;
    if(deviceID > countGpus -1)
    {
        std::cout<<"########## select error gpu id"<<" ##########"<<std::endl;
        cudaSetDevice(0);
        return;
    }
    cudaSetDevice(deviceID);
    //std::cout<<"########## select gpu id "<<deviceID<<" ##########"<<std::endl;
}



// These is necessary if we want to be able to write 1_GiB instead of 1.0_GiB.
// Since the return type is signed, -1_GiB will work as expected.
constexpr long long int operator"" _GiB(long long unsigned int val)
{
    return val * (1 << 30);
}
constexpr long long int operator"" _MiB(long long unsigned int val)
{
    return val * (1 << 20);
}
constexpr long long int operator"" _KiB(long long unsigned int val)
{
    return val * (1 << 10);
}

inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

static Logger gLogger{Logger::Severity::kINFO};
static LogStreamConsumer gLogVerbose{LOG_VERBOSE(gLogger)};
static LogStreamConsumer gLogInfo{LOG_INFO(gLogger)};
static LogStreamConsumer gLogWarning{LOG_WARN(gLogger)};
static LogStreamConsumer gLogError{LOG_ERROR(gLogger)};
static LogStreamConsumer gLogFatal{LOG_FATAL(gLogger)};

static void setReportableSeverity(Logger::Severity severity)
{
    gLogger.setReportableSeverity(severity);
    gLogVerbose.setReportableSeverity(severity);
    gLogInfo.setReportableSeverity(severity);
    gLogWarning.setReportableSeverity(severity);
    gLogError.setReportableSeverity(severity);
    gLogFatal.setReportableSeverity(severity);
    return;
}

static std::vector<std::string>readFolder(const std::string &image_path)
{
    std::vector<std::string> image_names;
    auto dir = opendir(image_path.c_str());

    if ((dir) != nullptr)
    {
        struct dirent *entry;
        entry = readdir(dir);
        while (entry)
        {
            auto temp = image_path + "/" + entry->d_name;
            if (strcmp(entry->d_name, "") == 0 || strcmp(entry->d_name, ".") == 0
                    || strcmp(entry->d_name, "..") == 0)
            {
                entry = readdir(dir);
                continue;
            }
            image_names.push_back(temp);
            entry = readdir(dir);
        }
    }
    return image_names;
}

static bool readTrtFile(const std::string &engineFile, //name of the engine file
                 nvinfer1::ICudaEngine *&engine)
{
    std::string cached_engine;
    std::fstream file;
    std::cout << "loading filename from:" << engineFile << std::endl;

    file.open(engineFile, std::ios::binary | std::ios::in);

    if (!file.is_open())
    {
        std::cout << "read file error: " << engineFile << std::endl;
        cached_engine = "";
    }

    while (file.peek() != EOF)
    {
        std::stringstream buffer;
        buffer << file.rdbuf();
        cached_engine.append(buffer.str());
    }
    file.close();
    nvinfer1::IRuntime *trtRuntime;

    trtRuntime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());
    engine = trtRuntime->deserializeCudaEngine(cached_engine.data(),cached_engine.size());
    std::cout << "deserialize done" << std::endl;

    return true;
}

static void onnxToTRTModel(const std::string &modelFile, // name of the onnx model
                    const std::string &filename,  // name of saved engine
                    nvinfer1::ICudaEngine *&engine, const int &BATCH_SIZE)
{
    // create the builder
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    auto config = builder->createBuilderConfig();

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    if (!parser->parseFromFile(modelFile.c_str(), static_cast<int>(gLogger.getReportableSeverity()))) {
        gLogError << "Failure while parsing ONNX file" << std::endl;
    }
    // Build the engine
    builder->setMaxBatchSize(BATCH_SIZE);
    config->setMaxWorkspaceSize(1_GiB);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);

    std::cout << "start building engine" << std::endl;
    engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build engine done" << std::endl;
    assert(engine);
    // we can destroy the parser
    parser->destroy();
    // save engine
    nvinfer1::IHostMemory *data = engine->serialize();
    std::ofstream file;
    file.open(filename, std::ios::binary | std::ios::out);
    std::cout << "writing engine file..." << std::endl;
    file.write((const char *) data->data(), data->size());
    std::cout << "save engine file done" << std::endl;
    file.close();
    // then close everything down
    network->destroy();
    builder->destroy();
}

static std::map<int, std::string> readImageNetLabel(const std::string &fileName)
{
    std::map<int, std::string> imagenet_label;
    std::ifstream file(fileName);
    if (!file.is_open())
    {
        std::cout << "read file error: " << fileName << std::endl;
    }
    std::string strLine;
    while (std::getline(file, strLine))
    {
        int pos1 = strLine.find(":");
        std::string first = strLine.substr(0, pos1);
        int pos2 = strLine.find_last_of("'");
        std::string second = strLine.substr(pos1 + 3, pos2 - pos1 - 3);
        imagenet_label.insert({atoi(first.c_str()), second});
    }
    file.close();
    return imagenet_label;
}

static std::map<int, std::string> readCOCOLabel(const std::string &fileName)
{
    std::map<int, std::string> coco_label;
    std::ifstream file(fileName);
    if (!file.is_open())
    {
        std::cout << "read file error: " << fileName << std::endl;
    }
    std::string strLine;
    int index = 0;
    while (std::getline(file, strLine))
    {
        coco_label.insert({index, strLine});
        index++;
    }
    file.close();
    return coco_label;
}

} // namespace bbwh

#endif // ENOCR_CPP_COMMON_COMMON_H_
