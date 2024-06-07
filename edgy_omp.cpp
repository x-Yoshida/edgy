#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
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
    long frames=LONG_MAX;
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
        
        if(args[i]=="-f" || args[i]=="--frames")
        {
            if(++i<args.size() && args[i][0]!='-')
            {
                flags.frames = std::stoi(args[i]);
                
            }
            else
            {
                std::cout<<"Didn't provided number of threads. Skipping...\n";
            }
            continue;

        }

        if(args[i]=="-s" || args[i]=="--show")
        {
            flags.show = true;
            continue;
        }
        if(args[i]=="-fo" || args[i]=="--file")
        {
            flags.file_output = true;
            continue;
        }
    }
}

void DetectEdges(Color* input,Color* output,const ImgData& data,const int& threshold)
{

    #pragma omp parallel for default(none) shared(input,output,data,threshold)
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

void Setup(const std::string path,const AppFlags &flags)
{
    cv::VideoCapture cap(path);


    // Remains of testing library
    // int fps=cap.get(cv::CAP_PROP_FPS);
    // int fc=cap.get(cv::CAP_PROP_FRAME_COUNT);
    // int duration=std::ceil((float)fc/(float)fps);
    // std::cout << fps << std::endl;
    // std::cout << fc << std::endl;
    // std::cout << duration << std::endl;
    // std::cout << cap.get(cv::CAP_PROP_FRAME_WIDTH) << std::endl;
    // std::cout << cap.get(cv::CAP_PROP_FRAME_HEIGHT ) << std::endl;


    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        exit(-1);
    }
    
    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat frame;
    
    //Captures first frame
    cap>>frame;

    //Step stores number of bytes that one row takes, rows stores how many rows matrix has;
    int colorBytes = frame.step * frame.rows;
    ImgData data = {frame.cols,frame.rows,frame.step};
    
    Color *d_input = new Color[frame.cols * frame.rows];
    Color *d_output = new Color[frame.cols * frame.rows];
    // #pragma omp parallel
    // {
    //     std::cout<< "Thread: " << omp_get_thread_num() << std::endl;
    // }
    long i=0;    

    while(i<flags.frames && !frame.empty())
    {
        memcpy(d_input,frame.ptr(),colorBytes);

        DetectEdges(d_input,d_output,data,flags.threshold);

        cap>>frame;
        i++;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate the elapsed time
    std::chrono::duration<double> elapsed = end - start;


    // std::cout << "Frames processed: " << i << std::endl;
    std::cout << "Execution time: " << elapsed.count() << std::endl;
    
    //Not doing output because we only mesure how long algorithm takes (Probably will do output in free time)

    // memcpy(output.ptr(),d_output,colorBytes);
    delete[] d_input;
    delete[] d_output;
}


int main(int argc, char** argv)
{
    std::vector<std::string> args = MakeArgvIntoString(argc,argv);
    
    AppFlags flags;
    
    if(args.size()<2)
    {
        std::cout << "Usage " << args[0] << " path/to/vid" << std::endl;
    }

	std::string path = args[1];

    if(args.size()>2)
    {
        HandleFlags(args, flags);        
    }
    

    Setup(path,flags);


    return EXIT_SUCCESS;
}

