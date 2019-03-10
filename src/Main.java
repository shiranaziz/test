import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.awt.*;
import java.io.File;
import java.security.InvalidParameterException;
import java.util.function.ToDoubleBiFunction;

import static org.opencv.core.CvType.*;

public class Main {

    static final String fixedImgPath = "fixed.png";
    static final int grayscaleMaxVal = 255;
    static final int numInputParams = 5;
    static final int imagePathNumArg = 0;
    static final int maskPathNumArg = 1;
    static final int normNumArg = 2;
    static final int epsilonNumArg = 3;
    static final int connectivityTypeNumArg = 4;

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        if (args.length != numInputParams)
        {
            System.out.println("wrong number of input args. should be: " +
                    "<imagePath> <maskPath> <z> <eps> <pixelConnectivity>");
            return;
        }

        String originalImagePath = args[imagePathNumArg];
        File originalImageFile = new File(originalImagePath);
        if (!originalImageFile.exists() || !originalImageFile.isFile())
        {
            System.out.println("original image doesn't exist");
            return;
        }
        String maskPath = args[maskPathNumArg];
        File maskFile = new File(maskPath);
        if (!maskFile.exists() || !maskFile.isFile())
        {
            System.out.println("mask image doesn't exist");
            return;
        }
        int z =  Integer.parseInt(args[normNumArg]);
        float epsilon = Float.parseFloat(args[epsilonNumArg]);
        HoleFiller.PixelConnectivity connectivity = ParseConnectivity(args[connectivityTypeNumArg]);

        Mat img = Imgcodecs.imread(originalImagePath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat mask = Imgcodecs.imread(maskPath, Imgcodecs.IMREAD_GRAYSCALE);

//        Mat fixedImg = HoleFiller.Run(MergeImageAndMask(img, mask), DefaultWeightFunc(z, epsilon), connectivity);
//        Mat fixedImg = HoleFiller.RunFast(MergeImageAndMask(img, mask), DefaultWeightFunc(z, epsilon), connectivity);
        Mat fixedImg = HoleFiller.RunConv(MergeImageAndMask(img, mask), DefaultWeightFunc(z, epsilon), connectivity);

        fixedImg.convertTo(fixedImg, CV_8UC1, grayscaleMaxVal);
        Imgcodecs.imwrite(fixedImgPath,fixedImg);
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
        img.convertTo(imgWithHole, CV_32F, 1.0/grayscaleMaxVal);
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
    private static ToDoubleBiFunction<Point, Point> DefaultWeightFunc(int z , float epsilon){
        return (u,v) -> {
            double norm = Math.sqrt(Math.pow(u.x-v.x,2) + Math.pow(u.y-v.y,2));
            return (1/(Math.pow(norm,z)+ epsilon));
        };
    }
}

