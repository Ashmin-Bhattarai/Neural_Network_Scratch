#include <iostream>
#include <math.h>

using namespace std;

double rand_seed = time(0);


class Matrix
{
    private:
        double *temp;

    public:
        double *mat;
        int row , col;


        // empty constructor
        Matrix(){};
    
        // constructor initializer
        void init(int m, int n, int initializer=0, bool isRandom=false)
        {
            mat  = new double[m*n];
            row = m;
            col = n;
            
            if(isRandom == false)
            {
                for (int i = 0; i < m * n; i++)
                {
                    *(mat + i) = initializer;
                }
            }
            else
            {   
                srand(rand_seed);
                for (int i = 0; i < m * n; i++)
                {
                    *(mat + i) = rand()/double(RAND_MAX);
                }
            }
        }


        // initilizes if argument is pass with initilzing
        Matrix(int m, int n, int initializer=0, bool isRandom=false)
        {
            init(m, n, initializer, isRandom);
        }


        Matrix operator * (const Matrix &mat2)
        {
            int m = this->row, n = this->col, o = mat2.col;
            
            // initializing final matrix
            Matrix ans(m,o);            
            temp = new double[this->row*mat2.col];
            
            for(int i=0; i<m*o; i++)
            {
                *(temp + i) = 0;
            }

            //matrice multiplication 
            for(int i=0; i<m; i++)
            {
                for(int j=0; j<n; j++)
                {
                    for(int k=0; k<o; k++)
                    {
                        *(temp + i*o + k) = *(this->mat + i*n + j) * *(mat2.mat + j*o + k) + *(temp + i*o + k);
                    }
                }
            }

            ans.mat = temp;
            
            return ans;
        }

        
        Matrix operator + (const Matrix &mat2)
        {
            int m = this->row, n = this->col, o = mat2.col;
            
            Matrix ans(m,n);
            temp = new double[m*n];
            
            for(int i=0; i<m*n; i++)
            {
                *(temp + i) = *(this->mat + 1) + *(mat2.mat + i);
            }
            
            ans.mat = temp;

            return ans;
        }


        void printMatrix()
        {
            for(int i=0; i<row; i++)
            {
                for(int j=0; j<col; j++)
                {
                    cout << *(mat + i*col + j) << "\t";
                }
                cout << endl;
            }
        } 
        

        Matrix transform()
        {
            Matrix trans(col, row);
            for(int i=0; i<row; i++)
            {
                for(int j=0; j<col; j++)
                {
                    *(trans.mat + j*row + i) = *(mat + i*col + j);
                }
            }
            return trans;
        }

        Matrix* multiply(Matrix mat1, Matrix mat2)
        {
            for(int i=0; i<row*col; i++)
            {
                *(mat + i) = *(mat1.mat + i) * *(mat2.mat + i);
            }
            return this;
        }

        Matrix getRow(int r)
        {
            Matrix rowM(col, 1);
            for(int i=0; i<col; i++)
            {
                *(rowM.mat + i) = *(mat + r*col + i);
            }
            return rowM;
        }
        
    
};






class Layer{

    public:
        Matrix input;
        Matrix output;
        Layer()
        {

        }
};




class Dense : private Layer
{
    private:
        Matrix weigths;
        Matrix biases;
        Matrix input;
        Matrix output;

    public:
        Dense(int input_size, int output_size)
        {
            weigths.init(output_size, input_size, 0, true);
            biases.init(output_size, 1, 0, true);
            output.init(output_size, 1, 0, false);
        }

        Matrix forward(Matrix inp)
        {
            input = inp;
            output = (weigths * input) + biases;
            return output;
        }

        Matrix backward(Matrix output_gradient, double lr=0.01)
        {
            Matrix weight_gradient = output_gradient * input.transform();
            Matrix input_gradient = weigths.transform() * output_gradient;
            for(int i=0; i<weigths.row*weigths.col; i++)
            {
                *(weigths.mat + i) -= lr * *(weight_gradient.mat + i);
            }
            for(int i=0; i<biases.row*biases.col; i++)
            {
                *(biases.mat + i) -= lr * *(output_gradient.mat + i);
            }

            return input_gradient;
        }
};


class Activation : private Layer
{
    public:
        Matrix *activation(Matrix);
        Matrix *activation_prime(Matrix);


        Matrix TanH(Matrix mat1)
        {
            Matrix mat2(mat1.row, mat1.col);
            for(int i=0; i<mat1.row*mat1.col; i++)
            {
                *(mat2.mat + i) = tanh(*(mat1.mat + i));
            }
            return mat2;
        }

        Matrix TanH_Prime(Matrix mat1)
        {
            Matrix mat2(mat1.row, mat1.col);
            for(int i=0; i<mat1.row*mat1.col; i++)
            {
                *(mat2.mat + i) = 1 - pow(*(mat1.mat + i), 2);
            }
            return mat2;
        }

        Matrix forward(Matrix inp)
        {
            input = inp;
            output = TanH(input);
            return output;
        }

        Matrix backward(Matrix output_gradient, double lr=0.01)
        {
            Matrix out(output_gradient.row, output_gradient.col);
            out.multiply(output_gradient, TanH_Prime(input));
            return out;
        }
};



double mse(Matrix y_true, Matrix y_pred)
{
    double mse = 0;
    for(int i=0; i<y_true.row*y_true.col; i++)
    {
        mse += pow(*(y_true.mat + i) - *(y_pred.mat + i), 2);
    }
    return mse/y_true.row*y_true.col;
}


Matrix mse_prime(Matrix y_true, Matrix y_pred)
{
    Matrix op_grad(y_true.row, y_true.col);
    
    for(int i=0; i<y_true.row*y_true.col; i++)
    {
        *(op_grad.mat + i) += 2 * (*(y_pred.mat + i) - *(y_true.mat + i));
    }
    *(op_grad.mat) /= y_true.row*y_true.col;
    return op_grad;
}


struct model{
    Dense dense_obj;
    Activation activation_obj;
    bool is_only_activation = false;
};

model network[] = {
        {Dense(2,3), Activation()},
        {Dense(3,1), Activation()}
    };

Matrix predict(Matrix input)
{
    // size of model
    int len_network = sizeof(network)/sizeof(model);
    Matrix output = input;

    for (int i=0; i<len_network; i++)
    {
        output = network[i].dense_obj.forward(output);
        if(!network[i].is_only_activation)
        {
            output = network[i].activation_obj.forward(output);
        }
    }
    return output;
}


void fit(Matrix x_train, Matrix y_train, int epochs = 50,  double lr=0.01)
{
    // size of model
    int len_network = sizeof(network)/sizeof(model);
    Matrix output;
    double error = 0.0, ter;

    for(int i=0; i<epochs; i++)
    {
        error = 0.0;
        for(int j=0; j<x_train.row; j++)
        {
            Matrix input = x_train.getRow(j);
            Matrix target = y_train.getRow(j);
            
            
            // forward pass
            output = predict(input);

            // cout << "\n========\nBefore forward pass" << endl;
            // cout << "input" << endl;
            // input.printMatrix();
            // cout << "\ntarget" << endl;
            // target.printMatrix();
            // cout << "\noutput" << endl;
            // output.printMatrix();

            // error
            ter = mse(target, output);
            error += ter;
            // cout << "\nTemp Error: " << ter << endl;

            // backward pass
            Matrix grad = mse_prime(target, output);

            for(int k=len_network-1; k>=0; k--)
            {
                if(!network[k].is_only_activation)
                {
                    grad = network[k].activation_obj.backward(grad, lr);
                }
                grad = network[k].dense_obj.backward(grad, lr);
            }

        }
        // cout << "Epoch: " << i+1 << " Error Before : " << error << endl;
        error /= x_train.row;
        cout << "Epoch: " << i+1 << " Error: " << error << endl;
    }
}




int main() {
    cout << "===============Neural Network================" << endl;
    cout << "=============================================" << endl;

    int trainX_row = 4;
    int trainX_col = 2;
    int trainY_row = 4;
    int trainY_col = 1;
    int testX_row = 1;
    int testX_col = 2;
    int testY_row = 1;
    int testY_col = 1;

    int epochs = 15;
    double lr = 0.1;


//  trainig data
    double X[trainX_row][trainX_col] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    double Y[trainY_row][trainY_col] = {{0.0},{1.0},{1.0},{0.0}};


//  trainig data matrix
    Matrix x_train(trainX_row, trainX_col);
    Matrix y_train(trainY_row, trainY_col);

    for(int i=0; i<trainX_row; i++)
    {
        for(int j=0; j<trainX_col; j++)
        {
            *(x_train.mat + i*trainX_col + j) = X[i][j];
        }
    }

    for(int i=0; i<trainY_row; i++)
    {
        for(int j=0; j<trainY_col; j++)
        {
            *(y_train.mat + i*trainY_col + j) = Y[i][j];
        }
    }


    // test data
    double X_test[testX_row][testX_col] = {{0.0, 0.0}};
    double Y_test[testY_row][testY_col] = {{0.0}};

    // test data matrix
    Matrix x_test(testX_row, testX_col);
    for(int i=0; i<testX_row; i++)
    {
        for(int j=0; j<testX_col; j++)
        {
            *(x_test.mat + i*testX_col + j) = X_test[i][j];
        }
    }

    Matrix y_test(testY_row, testY_col);
    for(int i=0; i<testY_row; i++)
    {
        for(int j=0; j<testY_col; j++)
        {
            *(y_test.mat + i*testY_col + j) = Y_test[i][j];
        }
    }

    cout << "===============Traning Started================" << endl;
    fit(x_train, y_train, epochs, lr);
    cout << "===============Traning Finished================" << endl;

    cout << "===============Testing Started================" << endl;
    Matrix output = predict(x_test.transform());
    cout << "Predicted output: " << endl;
    x_test.printMatrix();
    output.printMatrix();
    cout << "Test Error: " << mse(y_test, output) << endl;
    cout << "===============Testing Finished================" << endl;

    return 0;
}