import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.function.ToDoubleBiFunction;

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
        return new Mat();
    }

    private ArrayList<int[]> FindBoundary(Mat img){
        ArrayList<int[]> boundary = new ArrayList<>();
        if (this.connectivity == PixelConnectivity.FOUR){
            //something
        }
        else {
            //something
        }
        //TODO: write code...
        return boundary;
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
