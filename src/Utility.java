import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.util.function.Function;

import static org.opencv.core.CvType.CV_32F;

public class Utility {

    int z;
    float epsilon;
    Mat imgWithHole;

    public Utility(String originalImagePath, String maskPath,
                   int z, float epsilon){
        this.z = z;
        this.epsilon = epsilon;

        Mat img = Imgcodecs.imread(originalImagePath, Imgcodecs.IMREAD_GRAYSCALE);
        Mat mask = Imgcodecs.imread(maskPath, Imgcodecs.IMREAD_GRAYSCALE);
        MergeImageAndMask(img, mask);
//        System.out.println(imgWithHole.dump());
        HoleFiller holeFiller = new HoleFiller(Main::WeightFuncWrapper ,imgWithHole);
    }

    private void MergeImageAndMask(Mat img, Mat mask){
        this.imgWithHole = new Mat();
        img.convertTo(this.imgWithHole, CV_32F, 1.0/255);

        for (int i=0; i<img.rows(); i++){
            for (int j=0; j<img.cols(); j++){
                if (mask.get(i,j) [0] == 0){
                    this.imgWithHole.put(i,j,-1);
                }
            }
        }
    }

    public void Run(){
        System.out.println("running");
    }

    private static Function<int[], Float> WeightFuncWrapper(int[] u, int[] v, int z, float epsilon) {
        return Main::DefaultWeightFunc;
    }

    private static float DefaultWeightFunc(int[] u, int[] v, int z, float epsilon){
        double norm = Math.sqrt(Math.pow(u[0]-v[0],2) + Math.pow(u[1]-v[1],2));
        return (float) (1/(Math.pow(norm,z)+epsilon));
    }
}
