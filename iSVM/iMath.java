

public class iMath {
	public double dot(double[] a1, double[] a2) {
		if (a1.length != a2.length)
			throw new IllegalArgumentException(a1.length + " " + a2.length);
		double result = 0.0;
		for (int i = 0; i < a1.length; i++)
			result += a1[i] * a2[i];
		return result;
	}
	
	public double kernel(double[] a1, double[] a2) {
   /*
		double result = 0.0;
		for (int i = 0; i < a1.length; i++)
			result += Math.pow(a1[i] - a2[i], 2);
		return Math.exp(-result / 2);
     */ 
      return dot(a1, a2); 
	}
	
	public void aborted(int[] target, double[][] point, double[] u, double[] w) {
		/**
		 * to convert L coefficient to weight matrix
		 */
		w = new double[target.length];
		for (int i = 0; i < target.length; i++) {
			for (int j = 0; j < point[0].length; j++) {
				w[j] += u[i] * target[i] * point[i][j];
			}
		}
	}
}
