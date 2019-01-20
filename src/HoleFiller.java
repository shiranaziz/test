import org.opencv.core.Mat;

import java.util.ArrayList;
import java.util.function.ToDoubleBiFunction;

class HoleFiller {
    private ToDoubleBiFunction weightFunc;

    HoleFiller(ToDoubleBiFunction<int [], int[]> weightFunc){
        this.weightFunc = weightFunc;
    }

    Mat Run(Mat img){
        ArrayList<int[]> boundary = FindBoundary(img);
        //TODO: foreach u in H, 
        return new Mat();
    }

    private ArrayList<int[]> FindBoundary(Mat img){
        ArrayList<int[]> boundary = new ArrayList<>();
        //TODO: write code...
        return boundary;
    }
    
    private float RecoverPixel(int[] pixel, ArrayList<int[]> boundary, Mat img){
        float normalization = CalcNormalization(pixel, boundary);
        float result = 0;
        for (int[] v: boundary) {
            result += this.weightFunc.applyAsDouble(pixel,v) * img.get(v[0],v[1])[0];
        }
        return result/normalization;
    }
    
    private float CalcNormalization(int[] pixel, ArrayList<int[]> boundary){
        float normalization = 0;
        for (int[] v: boundary) {
            normalization += this.weightFunc.applyAsDouble(pixel,v);
        }
        return normalization;
    }
}
