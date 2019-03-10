import org.opencv.core.Core;
import org.opencv.core.Mat;

import java.awt.*;
import java.util.*;
import java.util.function.ToDoubleBiFunction;

import static org.opencv.core.CvType.*;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

class HoleFiller {

    enum PixelConnectivity{
        FOUR,
        EIGHT
    }

    private static final int numRandPixels = 30;

    /**
     * apply the exact hole filling algorithm using straight forward implementation.
     * complexity: O(n^1.5)
     * @param img matrix representing an image with hole
     * @param weightFunc weight function
     * @param connectivity pixel connectivity of the boundary
     * @return matrix representing a restored image (filled hole)
     */
    static Mat Run(Mat img, ToDoubleBiFunction<Point, Point> weightFunc, PixelConnectivity connectivity) {
        ArrayList<Point> hole = FindHole(img);
        ArrayList<Point> boundary = FindBoundary(connectivity, hole);
        Mat restoredImg = new Mat();
        img.copyTo(restoredImg);
        for (Point p : hole) {
            restoredImg.put(p.x, p.y, RecoverPixel(p, boundary, img, weightFunc));
        }
        return restoredImg;
    }

    /**
     * apply an approximate hole filling algorithm. this algorithm is stochastic:
     * uses a fixed number of randomly chosen boundary points to extrapolate from.
     * complexity: O(n)
     * @param img matrix representing an image with hole
     * @param weightFunc weight function
     * @param connectivity pixel connectivity of the boundary
     * @return matrix representing a restored image (filled hole)
     */
    static Mat RunFast(Mat img, ToDoubleBiFunction<Point, Point> weightFunc, PixelConnectivity connectivity) {

        ArrayList<Point> hole = FindHole(img);
        ArrayList<Point> boundary = FindBoundary(connectivity, hole);

        ArrayList<Point> boundarySubset = new ArrayList<>();
        Random rand = new Random();
        for (int i = 0; i < numRandPixels; i++) {
            int indx = rand.nextInt(boundary.size());
            boundarySubset.add(boundary.get(indx));
            boundary.remove(indx);
        }

        Mat restoredImg = new Mat();
        img.copyTo(restoredImg);
        for (Point p : hole) {
            restoredImg.put(p.x, p.y, RecoverPixel(p, boundarySubset, img, weightFunc));
        }
        return restoredImg;
    }

    /**
     * apply the exact hole filling algorithm using convolution implementation.
     * complexity: O(nlogn)
     * @param img matrix representing an image with hole
     * @param weightFunc weight function
     * @param connectivity pixel connectivity of the boundary
     * @return matrix representing a restored image (filled hole)
     */
    static Mat RunConv(Mat img, ToDoubleBiFunction<Point, Point> weightFunc, PixelConnectivity connectivity) {
        ArrayList<Point> hole = FindHole(img);
        ArrayList<Point> boundary = FindBoundary(connectivity, hole);
        Point center = CenterOfMass(boundary);
        int cropFaceLength = boundary.size() / 2;
        Point upperLeft = new Point(center.x - cropFaceLength / 2, center.y - cropFaceLength / 2);

        Mat boundaryMat = new Mat(cropFaceLength, cropFaceLength, CV_32F, new Scalar(0));
        Mat boundaryIndicatorMat = new Mat(cropFaceLength, cropFaceLength, CV_32F, new Scalar(0));
        for (Point p : boundary) {
            int x = p.x - upperLeft.x;
            int y = p.y - upperLeft.y;
            boundaryMat.put(x, y, img.get(p.x, p.y)[0]);
            boundaryIndicatorMat.put(x, y, 1);
        }

        Mat kernel = BuildKernel(weightFunc, cropFaceLength);

        Mat convMat = new Mat();
        Imgproc.filter2D(boundaryMat, convMat, -1, kernel);
        Mat normalizationMat = new Mat();
        Imgproc.filter2D(boundaryIndicatorMat, normalizationMat, -1, kernel);

        Mat restoredImgCrop = new Mat();
        Core.divide(convMat, normalizationMat, restoredImgCrop);

        Mat restoredImg = new Mat();
        img.copyTo(restoredImg);
        for (Point p : hole) {
            int origX = p.x - upperLeft.x;
            int origY = p.y - upperLeft.y;
            restoredImg.put(p.x, p.y, restoredImgCrop.get(origX, origY)[0]);
        }

        return restoredImg;
    }

    /**
     * find the center of a group of 2D points
     * @param points a list of points
     * @return center point
     */
    private static Point CenterOfMass(ArrayList<Point> points) {
        int numPoints = points.size();
        Point center = new Point();
        for (Point p : points) {
            center.translate(p.x, p.y);
        }
        center.setLocation(center.x/numPoints, center.y/numPoints);
        return center;
    }

    /**
     * generate a centered kernel matrix from a given weight function
     * @param weightFunc distance weight function
     * @param kernelSize size of kernel
     * @return kernel matrix
     */
    private static Mat BuildKernel(ToDoubleBiFunction<Point, Point> weightFunc, int kernelSize) {
        Mat kernel = new Mat(kernelSize, kernelSize, CV_32F);
        Point center = new Point(kernelSize / 2, kernelSize / 2);
        for (int i = 0; i < kernel.rows(); i++) {
            for (int j = 0; j < kernel.cols(); j++) {
                double val = weightFunc.applyAsDouble(center, new Point(i, j));
                kernel.put(i, j, val);
            }
        }
        return kernel;
    }

    /**
     * restore a single pixel in the hole using given boundary pixels
     * @param pixel pixel coordinates
     * @param boundary all boundary pixels
     * @param img image with hole
     * @param weightFunc wight function
     * @return estimated pixel value
     */
    private static float RecoverPixel(Point pixel, ArrayList<Point> boundary, Mat img,
                                      ToDoubleBiFunction<Point, Point> weightFunc) {
        float normalization = 0;
        float result = 0;
        for (Point v : boundary) {
            result += weightFunc.applyAsDouble(pixel, v) * img.get(v.x, v.y)[0];
            normalization += weightFunc.applyAsDouble(pixel, v);
        }
        return result / normalization;
    }

    /**
     * find the boundary of the hole
     * @param hole the hole we found at the original image
     * @param connectivity pixel connectivity
     * @return a list of boundary pixels
     */
    private static ArrayList<Point> FindBoundary(PixelConnectivity connectivity, ArrayList<Point> hole) {
        HashSet<Point> boundary = new HashSet<>();
        for (Point point: hole) {
            ArrayList<Point> connected = connectivity == PixelConnectivity.FOUR ?
                    GetFourNeighborhood(point) : GetEightNeighborhood(point);
            boundary.addAll(connected);
        }
        boundary.removeAll(hole);
        System.out.println(String.format("m = %d",boundary.size()));
        return new ArrayList<>(boundary);
    }

    /***
     * find the hole pixels
     * @param img image with hole
     * @return a list of hole pixels
     */
    private static ArrayList<Point> FindHole(Mat img) {
        ArrayList<Point> hole = new ArrayList<>();
        int n = 0;
        for (int i = 0; i < img.rows(); i++) {
            for (int j = 0; j < img.cols(); j++) {
                if (img.get(i, j)[0] == -1) {
                    n++;
                    hole.add(new Point(i,j));
                }
            }
        }
        System.out.println(String.format("n = %d",n));
        return hole;
    }

    /**
     * get the four connected pixels
     * @param p index point
     * @return list of connected pixels
     */
    private static ArrayList<Point> GetFourNeighborhood(Point p){
        ArrayList<Point> lst = new ArrayList<>(4);
        lst.add(new Point(p.x - 1, p.y));
        lst.add(new Point(p.x + 1, p.y));
        lst.add(new Point(p.x, p.y - 1));
        lst.add(new Point(p.x, p.y + 1));
        return  lst;
    }

    /**
     * get the eight connected pixels
     * @param p index point
     * @return list of connected pixels
     */
    private static ArrayList<Point> GetEightNeighborhood(Point p){
        ArrayList<Point> lst = new ArrayList<>(8);
        lst.addAll(GetFourNeighborhood(p));
        lst.add(new Point(p.x - 1, p.y - 1));
        lst.add(new Point(p.x - 1, p.y + 1));
        lst.add(new Point(p.x + 1, p.y - 1));
        lst.add(new Point(p.x + 1, p.y + 1));
        return lst;
    }
}
