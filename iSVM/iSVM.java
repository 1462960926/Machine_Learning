

import java.util.HashSet;
import java.util.Iterator;
import java.util.Random;
import net.sf.javaml.core.*;

/**
 * SMO algorithm, 
 * John C. Platt, 
 * Sequential Minimal Optimization:
   A Fast Algorithm for Training Support Vector Machines
 * @author Administrator
 *
 */
public class iSVM extends iMath {

	private int size;
	private int noAttributes;
	private int[] target;
	private double[][] point;
	private double[] u;
	private double b;
	private final double C = 1; // Initialize C to 1 //
	private final double tol = 0.01;
	private final double eps = 1e-3;
	private double[][] kernel; 
	private double[] E_cache;
	private HashSet<Integer> boundAlpha;
	Random random;
	private double[] w;
	
	public iSVM(Dataset data) {
		this.size = data.size();
		this.noAttributes = data.noAttributes();
		target = new int[size];
		point = new double[size][noAttributes];
		u = new double[size];
		b = 0.0;
		kernel = new double[size][size];
		E_cache = new double[size];
		boundAlpha = new HashSet<Integer>();
		random = new Random();
		{
			int i = 0;
			for (Instance instance : data) {
				if (instance.classValue().toString().trim().equals("1"))
					target[i] = 1;
				else 
					target[i] = -1;
				for (int j = 0; j < noAttributes; j++) {
					point[i][j] = instance.value(j);
				}
				E_cache[i] = getE(i);
				i++;
			}
		}
      for (int i = 0; i < size; i++) {
         for (int j = 0; j < size; j++) {
   			kernel[i][j] = kernel(point[i], point[j]);
         }
      }
		w = new double[noAttributes];
	}
	
	private boolean takeStep(int i1, int i2) {
		if (i1 == i2) return false;
		double alph1 = u[i1];
		double alph2 = u[i2];
		int y1 = target[i1];
		int y2 = target[i2];
		double E1 = getE(i1);
		double E2 = getE(i2);
		int s = y1 * target[i2];
		
		double L, H; 
		if (y1 != target[i2]) {
			L = Math.max(0, u[i2] - u[i1]);
			H = Math.min(C, C + u[i2] - u[i1]);
		} else {
			L = Math.max(0, u[i2] + u[i1] - C);
			H = Math.min(C, u[i2] + u[i1]);
		}
		if (L == H && (alph1 != 0 || alph2 != 0)) return false;
		
		double k11 = kernel(point[i1], point[i1]);
		double k12 = kernel(point[i1], point[i2]);
		double k22 = kernel(point[i2], point[i2]);
		double eta = k11 + k22 - 2 * k12;
		double a1, a2;
		double a2c; //** new **//
		if (eta > 0) {
			// double E2 = dot(u, point[i2]) - target[i2];
			a2c = u[i2] + target[i2] * (E1 - E2) / eta;//***//
			if (a2c < L) a2 = L;
			else if (a2c > H) a2 = H;
			else a2 = a2c;
			// u[i2] = a2;
		} else {
			double f1 = y1 * (E1 + b) - alph1 * kernel(point[i1], point[i1]) - s * alph2 * kernel(point[i1], point[i2]);
			double f2 = y2 * (E2 + b) - s * alph1 * kernel(point[i1], point[i2]) - alph2 * kernel(point[i2], point[i2]);
			double L1 = alph1 + s * (alph2 - L);
			double H1 = alph1 + s * (alph2 - H);
			double Lobj = L1 * f1 + L * f2 + 1.0 / 2 * L1 * L1 * k11 + 1.0 / 2 * L * L * k22 + s * L * L1 * k12;
			double Hobj = H1 * f1 + H * f2 + 1.0 / 2 * H1 * H1 * k11 + 1.0 / 2 * H * H * k22 + s * H * H1 * k12;
			if (Lobj < Hobj - eps)
				a2 = L;
			else if (Lobj > Hobj + eps)
				a2 = H;
			else 
				a2 = alph2;
			a2c = alph2;
		}
		if (Math.abs(a2 - alph2) < eps * (a2 + alph2 + eps))
			return false;
		a1 = alph1 + s * (alph2 - a2);
		/**
		 * Update threshold to reflect change in Lagrange multipliers
		 */
		double b1 = b - (E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - a2c) * k12);
		double b2 = b - (E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - a2c) * k22);
		if (0 < a1 && a1 < C)
			b = b1;
		else if (0 < a2 && a2 < C)
			b = b2;
		else 
			b = (b1 + b2) / 2;
		/**
		 * Update weight vector to reflect change in a1 & a2, if SVM is linear
		 */
		u[i1] = a1;
		u[i2] = a2;
		/**
		 * Update error cache using new Lagrange multipliers
		 */
		for (int i = 0; i < E_cache.length; i++) {
			E_cache[i] = getE(i);
		}
		/**
		 * Store a1 in the alpha array
		 */
		if (a1 > 0 && a1 < C)
			boundAlpha.add(i1);
		/**
		 * Store a2 in the alpha array
		 */
		if (a2 > 0 && a2 < C)
			boundAlpha.add(i2);
		return true;
	}
	
	private int examineExample(int i2) {
		int y2 = target[i2];
		double alph2 = u[i2];
		double E2 = getE(i2);
		double r2 = E2 * y2;
		if ((r2 < -tol && alph2 < C) 
				|| (r2 > tol && alph2 > 0)) {
			int i1;
			if (this.boundAlpha.size() > 0) {
				i1 = findMax(E2, this.boundAlpha);
			} else {
				i1 = RandomSelect(i2);
			}
			if (i1 != -1 && takeStep(i1, i2))
				return 1;
		}
		return 0;
	}
	
	public void train() {
		int numChanged = 0;
		int examineAll = 1;
		while (examineAll < 10) {
			if (numChanged == 0) {
            numChanged = 0;
				for (int i2 = 0; i2 < point.length; i2++) {
					numChanged += examineExample(i2);
				}
			} else {
            numChanged = 0;
				for (int i2 = 0; i2 < point.length; i2++) {
					if (u[i2] != 0 && u[i2] != C)
						numChanged += examineExample(i2);
				}
			}
		   if (numChanged == 0)
				examineAll = examineAll + 1;
         else 
            examineAll = 0;
		}
		
	}
	
	public double predict(Instance instance) {
		double[] x = new double[noAttributes];
		for (int i = 0; i < noAttributes; i++)
			x[i] = instance.value(i);
		double result = 0.0;
		for (int j = 0; j < target.length; j++) {
			result += target[j] * u[j] * kernel(x, point[j]);
		}
		result += b;
		return result;
	}
	
	private double f(int j) {
		double sum = 0.0;
		for (int i = 0; i < point.length; i++) {
			sum += u[i] * target[i] * kernel[i][j];
		}
		return sum + this.b;
	}
	
	private double getE(int i) {
		return f(i) - (double) target[i];
	}
	
	private int findMax(double Ei, HashSet<Integer> boundAlpha2) {
		double max = 0;
		int maxIndex = -1;
		for (int j : boundAlpha) {
			double Ej = getE(j);
			if (Math.abs(Ei - Ej) > max) {
				max = Math.abs(Ei - Ej);
				maxIndex = j;
			}
		}
		return maxIndex;
	}
	
	private int RandomSelect(int i) {
		int j;
		do {
			j = random.nextInt(point.length);
		} while(i == j);
		return j;
	}
}
