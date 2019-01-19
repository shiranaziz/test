public class Main {

    public static void main(String[] args) {
//        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        String originalImagePath = args[0];
        String maskPath = args[1];
        int z =  Integer.parseInt(args[2]);
        float epsilon = Float.parseFloat(args[3]);
        Utility utility = new Utility(originalImagePath,maskPath,z,epsilon);
        utility.Run();
        System.out.println("Finished");
    }
}
