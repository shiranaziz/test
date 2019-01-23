import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.security.InvalidParameterException;
import java.util.function.ToDoubleBiFunction;

import static org.opencv.core.CvType.*;

public class Main {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        String originalImagePath = args[0];
        String maskPath = args[1];
        int z =  Integer.parseInt(args[2]);
        float epsilon = Float.parseFloat(args[3]);
        HoleFiller.PixelConnectivity connectivity = ParseConnectivity(args[4]);

        Mat img = Imgcodecs.imread(originalImagePath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat mask = Imgcodecs.imread(maskPath, Imgcodecs.IMREAD_GRAYSCALE);

        Mat fixedImg = HoleFiller.Run(MergeImageAndMask(img, mask),DefaultWeightFunc(z, epsilon), connectivity);
//        Mat fixedImg = HoleFiller.RunFast(MergeImageAndMask(img, mask));

        Mat temp = new Mat();
        fixedImg.convertTo(temp,CV_8UC1,255);
        Imgcodecs.imwrite("fixed.png",temp);
        System.out.println("Finished");
    }

    /**
     * parse input argument for pixel connectivity into enum
     * @param arg the type of connectivity
     * @return parsed enum HoleFiller.PixelConnectivity
     */
    private static HoleFiller.PixelConnectivity ParseConnectivity(String arg) {
        if (arg.equals("4")){
            return HoleFiller.PixelConnectivity.FOUR;
        }
        else if (arg.equals("8")){
            return HoleFiller.PixelConnectivity.EIGHT;
        }
        else {
            throw new InvalidParameterException("invalid pixel connectivity: should be 4 or 8");
        }
    }

    /**
     * merge image and mask into single image with hole
     * @param img matrix representing the original image
     * @param mask matrix representing the hole
     * @return matrix representing an image with hole
     */
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

    /**
     * create the default weight function with specified z and epsilon parameters
     * @param z
     * @param epsilon
     * @return default weight function
     */
    private static ToDoubleBiFunction<int [], int[]> DefaultWeightFunc(int z , float epsilon){
        return (x,y) -> {
            double norm = Math.sqrt(Math.pow(x[0]-y[0],2) + Math.pow(x[1]-y[1],2));
            return (1/(Math.pow(norm,z)+ epsilon));
        };
    }
}

