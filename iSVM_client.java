
import java.io.File;
import java.io.IOException;

import net.sf.javaml.core.*;
import net.sf.javaml.tools.data.FileHandler;
public class iSVM_client {
	public static void main(String[] args) {
		Dataset data;
		try {
			data = FileHandler.loadDataset(new File("test_py"), 2, ",");
			iSVM svm = new iSVM(data);
			System.out.println(data);
			svm.train();
			Instance instance = new DenseInstance(new double[] {6, 130, 8});
			for (int i = 0; i < data.size(); i++)
            System.out.println(svm.predict(data.instance(i)));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
