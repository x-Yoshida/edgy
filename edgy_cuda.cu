#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

//Number of threads in block

//#define BLOCK_SIZE 256
//#define GRID_SIZE 1

int BLOCK_SIZE = 256;
long long GRID_SIZE = 1;


struct ImgData
{
    int width,height;
    unsigned long step;
};

struct Color
{
    unsigned char b,g,r;
    __device__ char GetDom()
    {
        int max = b;
        if(max<g)
        {
            max=g;
        }
        if(max<r)
        {
            return 'r';
        }
        if(max==g)
        {
            return 'g';
        }
        return 'b';
    }
    __device__ int Sum()
    {
        return r+b+g;
    }
};

struct Point
{
    int x,y;
    __device__ void GetNeighbours(int width,int height,Point* neighbours)
    {
        // |l|lu|u|ru|r|rd|d|ld (over engineered)
        // enum POS
        // {
        //     LEFT = 0b10000000,
        //     LEFTUP = 0b01000000,
        //     UP = 0b00100000,
        //     RIGHTUP = 0b00010000,
        //     RIGHT = 0b00001000,
        //     RIGHTDOWN = 0b00000100,
        //     DOWN = 0b00000010,
        //     LEFTDOWN = 0b00000001
        // };
        // unsigned char pos_flag = 0;

        if(x>0)
        {
            neighbours[0]={x-1,y};
            if(y>0)
            {
                neighbours[1]={x-1,y-1};
            }
            
            if(y<height-1)
            {
                neighbours[7]={x-1,y+1};
            }
        }

        if(x<width-1)
        {
            neighbours[4]={x+1,y};
            if(y>0)
            {
                neighbours[3]={x+1,y-1};
            }
            
            if(y<height-1)
            {
                neighbours[5]={x+1,y+1};
            }
        }

        if(y>0)
        {
            neighbours[2]={x,y-1};
        }
        
        if(y<height-1)
        {
            neighbours[6]={x,y+1};
        }
        

    }
};



std::vector<std::string> MakeArgvIntoString(int argc, char** argv)
{
    std::vector<std::string> args;
    for(int i=0;i<argc;i++)
    {
        args.push_back(argv[i]);
    }
    return args;
}

//threadIdx.x - thread index
//blockDim.x - number of threads in block
//blockIdx.x - block index in grid
//gridDim.x - grid size

__global__ void DetectEdges(Color* input,Color* output, ImgData data,int threshold)
{
    //Step so threads won't overlap
    int step = blockDim.x * gridDim.x;
    //Starting possition for thread
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    //long int size = data.height*data.width;
    for(int i=tid;i<data.height*data.width;i+=step)
    {
        bool done=false;
        Point pos = {.x=i % data.width, .y=i / data.width};
        Point neighbours[8];
        for(int j=0;j<8;j++)
        {
            neighbours[j]={-1,-1};
        }

        pos.GetNeighbours(data.width,data.height,neighbours);


        for(int j=0;j<8;j++)
        {
            if(neighbours[j].x==-1)
            {
                continue;
            }
            if(abs(input[i].Sum()-input[neighbours[j].y*data.width+neighbours[j].x].Sum())>=threshold)
            {
                output[i].r=255;
                output[i].g=255;
                output[i].b=255;
                done=true;
                break;
            }
        }        
        if(done)
        {
            continue;
        }
        output[i].r=0;
        output[i].g=0;
        output[i].b=0;

    }

}

/*
void Menu(cv::Mat& input, cv::Mat& output,int threshold)
{
    //Step stores number of bytes that one row takes, rows stores how many rows matrix has;
    int colorBytes = input.step * input.rows;
    //int dimmensions = input.rows * input.cols; 


    Color *d_input;
    Color *d_output;
    
	// Allocate device memory
    cudaMalloc<Color>(&d_input,colorBytes);
    cudaMalloc<Color>(&d_output,colorBytes);

	// Copy data from OpenCV input image to device memory
    cudaMemcpy(d_input,input.ptr(),colorBytes,cudaMemcpyHostToDevice);
    
    ImgData data = {input.cols,input.rows,input.step};

    int fun = 0;
    
    std::cout << "[1] Edge " << std::endl;
    std::cout << "[2] Min " << std::endl;
    std::cout << "[3] Max " << std::endl;
    std::cout << "[4] Dom " << std::endl;
    std::cout << "Choose function 1-4: " ;
    //std::cin >> fun;
    fun = 1;
    switch (fun-1)
    {
    case 0:
        DiffTreshhold<<<GRID_SIZE,BLOCK_SIZE>>>(d_input,d_output,data,threshold);
        break;
    case 1:
        MinColor<<<GRID_SIZE,BLOCK_SIZE>>>(d_input,d_output,data);
        break;
    case 2:
        MaxColor<<<GRID_SIZE,BLOCK_SIZE>>>(d_input,d_output,data);
        break;
    case 3:
        DomOnly<<<GRID_SIZE,BLOCK_SIZE>>>(d_input,d_output,data);
        break;
    case 4:
        test_fun2<<<GRID_SIZE,BLOCK_SIZE>>>(d_input,d_output,data);
        break;
    case 5:
        /* code *//*
        break;
    case 6:
        /* code *//*
        break;
    
    default:
        break;
    }


    cudaError res = cudaDeviceSynchronize();
    if(res)
    {
        std::cout << "CUDA: " << cudaGetErrorName(res) << std::endl;
    }
    
    
    cudaMemcpy(output.ptr(),d_output,colorBytes,cudaMemcpyDeviceToHost);
    cudaFree(d_input);
    cudaFree(d_output);

}
*/

void Setup(const std::string path,const int &threshold)
{

    // //Step stores number of bytes that one row takes, rows stores how many rows matrix has;
    // int colorBytes = input.step * input.rows;
    // //int dimmensions = input.rows * input.cols; 

    cv::VideoCapture cap(path);
    int fps=cap.get(cv::CAP_PROP_FPS);
    int fc=cap.get(cv::CAP_PROP_FRAME_COUNT);
    int duration=std::ceil((float)fc/(float)fps);
    std::cout << fps << std::endl;
    std::cout << fc << std::endl;
    std::cout << duration << std::endl;
    std::cout << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    std::cout << cap.get(cv::CAP_PROP_FRAME_HEIGHT ) << std::endl;

    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        exit(-1);
    }
    
    cv::Mat frame;
    
    //Captures first frame
    cap>>frame;

    //Step stores number of bytes that one row takes, rows stores how many rows matrix has;
    int colorBytes = frame.step * frame.rows;
    ImgData data = {frame.cols,frame.rows,frame.step};
    
    // Color *d_input = new Color[frame.cols * frame.rows];
    // Color *d_output = new Color[frame.cols * frame.rows];

    Color *d_input;
    Color *d_output;

	// Allocate device memory
    cudaMalloc<Color>(&d_input,colorBytes);
    cudaMalloc<Color>(&d_output,colorBytes);
    int i=0;    

    while(!frame.empty())
    {
        //memcpy(d_input,frame.ptr(),colorBytes);
        // Copy data from OpenCV input image to device memory
        cudaMemcpy(d_input,frame.ptr(),colorBytes,cudaMemcpyHostToDevice);

        //DetectEdges(d_input,d_output,data,threshold);
        DetectEdges<<<GRID_SIZE,BLOCK_SIZE>>>(d_input,d_output,data,threshold);


        // cudaError res = cudaDeviceSynchronize();
        // if(res)
        // {
        //     std::cout << "CUDA: " << cudaGetErrorName(res) << std::endl;
        // }

        cap>>frame;
        i++;
    }
    std::cout << i << std::endl;
    //Not doing output because we only mesure how long algorithm takes (Probably will do output in free time)

    // cudaMemcpy(output.ptr(),d_output,colorBytes,cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}


int main(int argc, char** argv)
{
    std::vector<std::string> args = MakeArgvIntoString(argc,argv);
    int cudaDevices;
    //bool show = false;
    //bool file_output = false;
    int threshold = 100;
    cudaGetDeviceCount(&cudaDevices);
    //std::cout << "Cuda devices found: " << cudaDevices << std::endl;
    if(args.size()<2)
    {
        std::cout << "Usage " << args[0] << " path/to/img" << std::endl;
    }

	std::string path = args[1];

    if(args.size()>2)
    {
        cudaDeviceProp dev_prop;
        cudaGetDeviceProperties(&dev_prop,0);
        //std::cout << "Max Threads per Block: " << dev_prop.maxThreadsPerBlock << std::endl;
        //std::cout << "Max Blocks: " << dev_prop.maxBlocksPerMultiProcessor << std::endl;
        //std::cout << "Max Grid Size: " << dev_prop.maxGridSize[0]  << std::endl;
        //std::cout << "Max Grid Size (2D): " << dev_prop.maxGridSize[1]  << std::endl;
        //std::cout << "Max Grid Size (3D): " << dev_prop.maxGridSize[2]  << std::endl;
        for(int i = 2;i<argc;i++)
        {
            //std::cout << args[i] << std::endl;
            if(args[i]=="-t" || args[i]=="--threads" )
            {
                if(++i<argc && args[i][0]!='-')
                {
                    BLOCK_SIZE = std::stoi(args[i]);
                    if(BLOCK_SIZE>dev_prop.maxThreadsPerBlock)
                    {
                        std::cout<<"CUDA: Can't use " << BLOCK_SIZE << " threads in block, changed to " << dev_prop.maxThreadsPerBlock << " instead\n";
                        BLOCK_SIZE = dev_prop.maxThreadsPerBlock;
                    }
                }
                else
                {
                    std::cout<<"Didn't provided number of threads. Skipping...\n";
                }
                continue;
            }
            if(args[i]=="-th" || args[i]=="--threshold")
            {
                if(++i<argc && args[i][0]!='-')
                {
                    threshold = std::stoi(args[i]);
                    if(threshold>3*255)
                    {
                        std::cout<<"Max possible threshold " << 3*255 << std::endl;
                        threshold = 3*255;
                    }
                }
                else
                {
                    std::cout<<"Didn't provided number of threads. Skipping...\n";
                }
                continue;

            }
            if(args[i]=="-gs" || args[i]=="--gridsize" )
            {
                if(++i<argc && args[i][0]!='-')
                {
                    GRID_SIZE = std::stoi(args[i]);
                    if(GRID_SIZE>dev_prop.maxGridSize[1])
                    {
                        std::cout<<"CUDA: Can't use " << GRID_SIZE << " threads in block, changed to " << dev_prop.maxGridSize[1] << " instead\n";
                        GRID_SIZE = dev_prop.maxGridSize[1];
                    }
                }
                else
                {
                    std::cout<<"Didn't provided number of threads. Skipping...\n";
                }
                continue;
            }
        }
    }
    
    // Menu(input,output,threshold);

    Setup(path,threshold);

    return EXIT_SUCCESS;
}

//https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html