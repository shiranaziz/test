import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.function.ToDoubleBiFunction;

import static org.opencv.core.CvType.CV_32F;

public class Main {

    public static void main(String[] args) {
        String originalImagePath = args[0];
        String maskPath = args[1];
        int z =  Integer.parseInt(args[2]);
        float epsilon = Float.parseFloat(args[3]);
        Mat img = Imgcodecs.imread(originalImagePath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat mask = Imgcodecs.imread(maskPath, Imgcodecs.IMREAD_GRAYSCALE);
        HoleFiller holeFiller = new HoleFiller(DefaultWeightFunc(z, epsilon));
        Mat fixedImg = holeFiller.Run(MergeImageAndMask(img, mask));
        System.out.println("Finished");
    }

    static private Mat MergeImageAndMask(Mat img, Mat mask){
        Mat imgWithHole = new Mat();
        img.convertTo(imgWithHole, CV_32F, 1.0/255);

        for (int i=0; i<img.rows(); i++){
            for (int j=0; j<img.cols(); j++){
                if (mask.get(i,j) [0] == 0){
                    imgWithHole.put(i,j,-1);
                }
            }
        }
        return imgWithHole;
    }

    private static ToDoubleBiFunction<int [], int[]> DefaultWeightFunc(int z , float epsilon){
        return (x,y) -> {
            double norm = Math.sqrt(Math.pow(x[0]-y[0],2) + Math.pow(x[1]-y[1],2));
            return (1/(Math.pow(norm,z)+ epsilon));
        };
    }
}

