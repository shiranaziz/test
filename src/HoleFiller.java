import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.function.ToDoubleBiFunction;

import static org.opencv.core.CvType.*;

class HoleFiller {

    enum PixelConnectivity{
        FOUR,
        EIGHT
    }

    private ToDoubleBiFunction weightFunc;
    private PixelConnectivity connectivity;

    HoleFiller(ToDoubleBiFunction<int [], int[]> weightFunc, PixelConnectivity connectivity){
        this.weightFunc = weightFunc;
        this.connectivity = connectivity;
    }

    Mat Run(Mat img){
        ArrayList<int[]> boundary = FindBoundary(img);
        Mat restoredImg = new Mat();
        img.copyTo(restoredImg);
        for (int i=0; i<restoredImg.rows(); i++){
            for (int j=0; j<restoredImg.cols(); j++){
                if (restoredImg.get(i,j) [0] == -1){
                    float recoveredPixel = RecoverPixel(new int[]{i,j},boundary,img);
                    restoredImg.put(i,j,recoveredPixel);
                }
            }
        }
        return restoredImg;
    }

    Mat RunFast(Mat img){
        Mat restoredImgHoriz = new Mat();
        img.copyTo(restoredImgHoriz);
        for (int i=0; i<restoredImgHoriz.rows(); i++){
            for (int j=0; j<restoredImgHoriz.cols(); j++){
                if (restoredImgHoriz.get(i,j) [0] == -1){
                    float recoveredPixel = RecoverPixelFast(new int[]{i,j},restoredImgHoriz);
                    restoredImgHoriz.put(i,j,recoveredPixel);
                }
            }
        }
        Mat restoredImgVert = new Mat();
        img.copyTo(restoredImgVert);
        for (int j=0; j<restoredImgHoriz.cols(); j++){
            for (int i=0; i<restoredImgHoriz.rows(); i++){
                if (restoredImgVert.get(i,j) [0] == -1){
                    float recoveredPixel = RecoverPixelFast(new int[]{i,j},restoredImgVert);
                    restoredImgVert.put(i,j,recoveredPixel);
                }
            }
        }
        Mat temp = new Mat();
        Core.add(restoredImgHoriz,restoredImgVert,temp);
        Mat restoredImg = new Mat();
        temp.convertTo(restoredImg,CV_32F,0.5);
        return restoredImg;
    }

    private float RecoverPixelFast(int[] pixel, Mat img) {
        ArrayList<int[]> connected = GetEightNeighborhood(pixel[0],pixel[1]);
        float mean = 0;
        int count = 0;
        for (int[] point:connected) {
            float val = (float) img.get(point[0],point[1])[0];
            if (val != -1){
                mean += val;
                count++;
            }
        }
        return mean/count;
    }

    private ArrayList<int[]> FindBoundary(Mat img) {
        HashSet<int[]> boundary = new HashSet<>();
        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                if (img.get(i, j)[0] == -1) {
                    ArrayList<int[]> connected;
                    if (this.connectivity == PixelConnectivity.FOUR){
                        connected = GetFourNeighborhood(i,j);
                    }
                    else {
                        connected = GetEightNeighborhood(i,j);
                    }
                    connected.removeIf(p->img.get(p[0],p[1])[0] == -1);
                    boundary.addAll(connected);
                }
            }
        }
        return new ArrayList<>(boundary);
    }

    private ArrayList<int[]> GetFourNeighborhood(int i, int j){
        ArrayList<int[]> lst = new ArrayList<>(4);
        lst.add(new int[]{i-1,j});
        lst.add(new int[]{i+1,j});
        lst.add(new int[]{i,j-1});
        lst.add(new int[]{i,j+1});
        return  lst;
    }

    private ArrayList<int[]> GetEightNeighborhood(int i, int j){
        ArrayList<int[]> lst = new ArrayList<>(8);
        lst.addAll(GetFourNeighborhood(i,j));
        lst.add(new int[]{i-1,j-1});
        lst.add(new int[]{i-1,j+1});
        lst.add(new int[]{i+1,j-1});
        lst.add(new int[]{i+1,j+1});
        return lst;
    }
    
    private float RecoverPixel(int[] pixel, ArrayList<int[]> boundary, Mat img){
        float normalization = 0;
        float result = 0;
        for (int[] v: boundary) {
            result += this.weightFunc.applyAsDouble(pixel,v) * img.get(v[0],v[1])[0];
            normalization += this.weightFunc.applyAsDouble(pixel,v);
        }
        return result/normalization;
    }
}
