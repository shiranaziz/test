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

    /**
     * apply the exact hole filling algorithm
     * @param img matrix representing an image with hole
     * @param weightFunc
     * @param connectivity pixel connectivity of the boundary
     * @return matrix representing a restored image (filled hole)
     */
    static Mat Run(Mat img, ToDoubleBiFunction<int [], int[]> weightFunc, PixelConnectivity connectivity){
        ArrayList<int[]> boundary = FindBoundary(img, connectivity);
        Mat restoredImg = new Mat();
        img.copyTo(restoredImg);
        for (int i=0; i<restoredImg.rows(); i++){
            for (int j=0; j<restoredImg.cols(); j++){
                if (restoredImg.get(i,j) [0] == -1){
                    float recoveredPixel = RecoverPixel(new int[]{i,j},boundary,img,weightFunc,connectivity);
                    restoredImg.put(i,j,recoveredPixel);
                }
            }
        }
        return restoredImg;
    }

    /**
     * apply an approximate hole filling algorithm (faster then the exact solution)
     * @param img matrix representing an image with hole
     * @return matrix representing a restored image (filled hole)
     */
    static Mat RunFast(Mat img){
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

    /**
     * restore a pixel in the hole, using the exact algorithm
     * @param pixel pixel coordinates
     * @param boundary all boundary pixels
     * @param img image with hole
     * @param weightFunc
     * @param connectivity boundary pixel connectivity
     * @return estimated pixel value
     */
    private static float RecoverPixel(int[] pixel, ArrayList<int[]> boundary, Mat img, ToDoubleBiFunction<int [], int[]> weightFunc, PixelConnectivity connectivity){
        float normalization = 0;
        float result = 0;
        for (int[] v: boundary) {
            result += weightFunc.applyAsDouble(pixel,v) * img.get(v[0],v[1])[0];
            normalization += weightFunc.applyAsDouble(pixel,v);
        }
        return result/normalization;
    }

    /**
     * restore a pixel in the hole, using the approximate algorithm
     * @param pixel pixel coordinates
     * @param img image with hole
     * @return estimated pixel value
     */
    private static float RecoverPixelFast(int[] pixel, Mat img) {
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

    /**
     * find the boundary of the hole
     * @param img image with hole
     * @param connectivity pixel connectivity
     * @return a list of boundary pixels
     */
    private static ArrayList<int[]> FindBoundary(Mat img, PixelConnectivity connectivity) {
        HashSet<int[]> boundary = new HashSet<>();
        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                if (img.get(i, j)[0] == -1) {
                    ArrayList<int[]> connected;
                    if (connectivity == PixelConnectivity.FOUR){
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

    /**
     * get the four connected pixels
     * @param i row index
     * @param j column index
     * @return list of connected pixels
     */
    private static ArrayList<int[]> GetFourNeighborhood(int i, int j){
        ArrayList<int[]> lst = new ArrayList<>(4);
        lst.add(new int[]{i-1,j});
        lst.add(new int[]{i+1,j});
        lst.add(new int[]{i,j-1});
        lst.add(new int[]{i,j+1});
        return  lst;
    }

    /**
     * get the eight connected pixels
     * @param i row index
     * @param j column index
     * @return list of connected pixels
     */
    private static ArrayList<int[]> GetEightNeighborhood(int i, int j){
        ArrayList<int[]> lst = new ArrayList<>(8);
        lst.addAll(GetFourNeighborhood(i,j));
        lst.add(new int[]{i-1,j-1});
        lst.add(new int[]{i-1,j+1});
        lst.add(new int[]{i+1,j-1});
        lst.add(new int[]{i+1,j+1});
        return lst;
    }
}
