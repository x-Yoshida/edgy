#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <omp.h>

//OMP_NUM_THREADS important;

struct ImgData
{
    int width,height;
    unsigned long step;
};

struct Color
{
    unsigned char b,g,r;
    char GetDom()
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
    int Sum()
    {
        return r+b+g;
    }
};

struct Point
{
    int x,y;
    void GetNeighbours(int width,int height,Point* neighbours)
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

struct AppFlags
{
    bool show = false;
    bool file_output = false;
    int threshold = 100;
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

void HandleFlags(std::vector<std::string> &args,AppFlags &flags)
{
    for(int i = 2;i<args.size();i++)
        {
            //std::cout << args[i] << std::endl;
            // if(args[i]=="-t" || args[i]=="--threads" )
            // {
            //     if(++i<args.size() && args[i][0]!='-')
            //     {
            //         BLOCK_SIZE = std::stoi(args[i]);
            //         if(BLOCK_SIZE>dev_prop.maxThreadsPerBlock)
            //         {
            //             std::cout<<"CUDA: Can't use " << BLOCK_SIZE << " threads in block, changed to " << dev_prop.maxThreadsPerBlock << " instead\n";
            //             BLOCK_SIZE = dev_prop.maxThreadsPerBlock;
            //         }
            //     }
            //     else
            //     {
            //         std::cout<<"Didn't provided number of threads. Skipping...\n";
            //     }
            //     continue;
            // }
            if(args[i]=="-th" || args[i]=="--threshold")
            {
                if(++i<args.size() && args[i][0]!='-')
                {
                    flags.threshold = std::stoi(args[i]);
                    if(flags.threshold>3*255)
                    {
                        std::cout<<"Max possible threshold " << 3*255 << std::endl;
                        flags.threshold = 3*255;
                    }
                }
                else
                {
                    std::cout<<"Didn't provided number of threads. Skipping...\n";
                }
                continue;

            }
            // if(args[i]=="-gs" || args[i]=="--gridsize" )
            // {
            //     if(++i<args.size() && args[i][0]!='-')
            //     {
            //         GRID_SIZE = std::stoi(args[i]);
            //         if(GRID_SIZE>dev_prop.maxGridSize[0])
            //         {
            //             std::cout<<"CUDA: Can't use " << GRID_SIZE << " threads in block, changed to " << dev_prop.maxGridSize[0] << " instead\n";
            //             GRID_SIZE = dev_prop.maxGridSize[0];
            //         }
            //     }
            //     else
            //     {
            //         std::cout<<"Didn't provided number of threads. Skipping...\n";
            //     }
            //     continue;
            // }
            if(args[i]=="-s" || args[i]=="--show")
            {
                flags.show = true;
                continue;
            }
            if(args[i]=="-nf" || args[i]=="--nofile")
            {
                flags.file_output = false;
                continue;
            }
        }
}

void DiffTreshhold(Color* input,Color* output, ImgData data,int threshold)
{
    //Step so threads won't overlap
    //Starting possition for thread
    
    //#pragma omp parallel
    {

        //int step = omp_get_num_threads();
        //std::cout << "step: " <<  step << std::endl;
        //int tid = omp_get_thread_num();

        //long int size = data.height*data.width;
        #pragma omp parallel for
        for(int i=0;i<data.height*data.width;i+=1)
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

}

void Menu(cv::Mat& input, cv::Mat& output,const int &threshold)
{
    //Step stores number of bytes that one row takes, rows stores how many rows matrix has;
    int colorBytes = input.step * input.rows;
    
    Color *d_input = new Color[input.cols * input.rows];
    Color *d_output = new Color[input.cols * input.rows];;
    
	// Allocate device memory
    //cudaMalloc<Color>(&d_input,colorBytes);
    //cudaMalloc<Color>(&d_output,colorBytes);

	// Copy data from OpenCV input image to device memory
    //cudaMemcpy(d_input,input.ptr(),colorBytes,cudaMemcpyHostToDevice);
    memcpy(d_input,input.ptr(),colorBytes);

    ImgData data = {input.cols,input.rows,input.step};

    DiffTreshhold(d_input,d_output,data,threshold);
    
    //cudaMemcpy(output.ptr(),d_output,colorBytes,cudaMemcpyDeviceToHost);
    memcpy(output.ptr(),d_output,colorBytes);
    delete[] d_input;
    delete[] d_output;
}

int main(int argc, char** argv)
{
    std::vector<std::string> args = MakeArgvIntoString(argc,argv);
    
    AppFlags flags;
    
    if(args.size()<2)
    {
        std::cout << "Usage " << args[0] << " path/to/img" << std::endl;
    }

	std::string imagePath = args[1];

    if(args.size()>2)
    {
        HandleFlags(args, flags);        
    }
    

    cv::Mat input = cv::imread(imagePath,cv::IMREAD_COLOR);
	cv::Mat output(input.rows,input.cols,CV_8UC3);


    if(flags.show)
        cv::imshow("Before",input);
    
    Menu(input,output,flags.threshold);

    if(flags.show)
        cv::imshow("After",output);

    // if(flags.file_output)
    //     cv::imwrite("out/output.png",output);
    
    if(flags.show)
        cv::waitKey();


    return EXIT_SUCCESS;
}

