
#include "BackProp.h"
// NeuralNet.cpp : Defines the entry point for the console application.

#include "backprop.h"
#include <fstream>
using namespace std;


double beta = 0.0005, alpha = 0.1, Thresh =  0.00001;
int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_label(int * labels)
{
	ifstream file ("t10k-labels.idx1-ubyte", ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
        int number_of_items = 0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_items,sizeof(number_of_items));
        number_of_items = reverseInt(number_of_items);
		for( int i = 0; i < number_of_items; ++i ){
			unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
			labels[i] = temp - 0x00 ;
		}
	}
}
void train_images(int *labels, CBackProp *bp, int num_iter)
{
	ifstream file ("t10k-images.idx3-ubyte", ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		for( int k = 0; k < num_iter; k++){
			file.seekg(0, ios_base::beg);
			magic_number=0;
			number_of_images=0;
			n_rows=0;
			n_cols=0;
			file.read((char*)&magic_number,sizeof(magic_number)); 
			magic_number= reverseInt(magic_number);
			file.read((char*)&number_of_images,sizeof(number_of_images));
			number_of_images= reverseInt(number_of_images);
			file.read((char*)&n_rows,sizeof(n_rows));
			n_rows= reverseInt(n_rows);
			file.read((char*)&n_cols,sizeof(n_cols));
			n_cols= reverseInt(n_cols);
			cout << n_rows << n_cols << endl;
			double img[784];
			for(int i=0;i < number_of_images;++i)
			{
				for(int r=0;r<n_rows;++r)
				{
					for(int c=0;c<n_cols;++c)
					{
						unsigned char temp=0;
						file.read((char*)&temp,sizeof(temp));
						if (file.good() ){
							img[r * 28 + c] = temp - 0x00;
						}
						else {
							cout << " bad file read" <<endl;
						}
					}
				}
				// read ith image
				double result[10];
				for(int k = 0; k < 10; k++)
					result[k] = 0;
				result[labels[i]] = 1;
				bp->bpgt(img, result); 
				
				/*
				 * End of one iteration. Parties start share their delta weights.
				 * get the intial weights at the beginning of this iteration
				 * get the delta weights at the end of this iteration
				 * calculate the average delta weight and apply these to initial weights.
				 */
			}
		}
		cout << "Iteration ends" << endl;
    }
}
void test_images(int *labels, CBackProp *bp)
{
	ifstream file ("t10k-images.idx3-ubyte", ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
		cout << n_rows << n_cols << endl;
		double img[784];
		int count = 0;
		for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
					if (file.good() ){
						img[r * 28 + c] = temp - 0x00;
					}
					else {
						cout << " bad file read" <<endl;
					}
                }
            }
			// read ith image
			bp->ffwd(img); 
			if( 1 - bp->Out(labels[i]) < 0.2 ){
				count++;
			}
			//cout << bp->Out(0) << " " << labels[i] << endl;
			/*
			if( bp->mse(&labels[i]) < Thresh) {
				cout << "Train complete " << endl;
				return ;
			}
			*/
        }
		cout << count / 10000.0 << endl;
		cout << "Iteration ends" << endl;
    }
}
int main(int argc, char* argv[])
{
	int labels[60000];
	read_label(labels);
	for (int i = 0; i < 1000; i++ )
		cout << labels[i];
	int numLayers = 3, lSz[3] = {784, 300, 10};

	// maximum no of iterations during training
	long num_iter = 100;

	// Creating the net
	CBackProp *bp = new CBackProp(numLayers, lSz, beta, alpha);
	cout<< endl <<  "Now training the network...." << endl;	
	train_images(labels, bp, num_iter);
	cout << "start test " << endl; 
	test_images(labels, bp);
	

	/*
	for (long i=0; i<num_iter ; i++)
	{
		
		bp->bpgt(data[i%80], &result[i%80]);

		if( bp->mse(&result[i%80]) < Thresh) {
			cout << endl << "Network Trained. Threshold value achieved in " << i << " iterations." << endl;
			cout << "MSE:  " << bp->mse(&result[i%80]) <<  endl <<  endl;
			break;
		}
		if ( i%(num_iter/10) == 0 )
			cout<<  endl <<  "MSE:  " << bp->mse(&result[i%80]) << "... Training..." << endl;

		if ( i == num_iter )
			cout << endl << i << " iterations completed..." << "MSE: " << bp->mse(&result[(i-1)%80]) << endl;

	}
	
	cout<< "Now using the trained network to make predctions on test data...." << endl << endl;	
	for (int i = 0 ; i < 80 ; i++ )
	{
		bp->ffwd(data[i]);
		cout << data[i][0]<< "  " << data[i][1]<< "  "  << bp->Out(0) << endl;
	}
	*/
	
	return 0;
}



