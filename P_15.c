/*
* Author:Pruthvi Suryadevara
* Email: pruthvi.suryadevara@tifr.res.in
* Description: Solving linear system LU decomposition 
* Compile using gcc P_15.c -lm -o P_15.out
*/

#include<stdio.h>
#include<stdlib.h>
#include <math.h>

double f(double y, double t)    // Defining the function f(y,t)=y'
{
  return( y -(t*t)+1);
}
double f_act(double t)          // Defining the actual solution
{
  return(((t+1)*(t+1))-(0.5*pow(2.718,t)));
}
double f_lim(double t,double h) // Defining the upper bound of error
{
  return(pow(2.718,t)*(pow(2.718,t)-1)*0.5*h);
}

int main()
{
  double t0,h,tn,y0,t;         // Initializing and defining problem paramaters
  FILE *fp;
  int st=remove("P_15.csv");
  fp = fopen("P_15.csv", "w+");  // Opening file to save output as csv
  t0=0;
  tn=2;
  y0=0.5;
  h=0.1;
  int n=((tn-t0)/h)+1;
  double y[n][5];
  y[0][0]=t0;
  y[0][1]=y0;
  fprintf(fp, "t,y,y_act,y_err,y_errlim \n");

  // Appling euler method and saving to csv file
  for(int i=0;i<n-1;i++)
    {
      y[i][2]=f_act(y[i][0]);
      y[i][3]=y[i][2]-y[i][1];
      y[i][4]=f_lim(y[i][0],h);
      y[i+1][1]=y[i][1]+(h*f(y[i][1],y[i][0]));
      y[i+1][0]=y[i][0]+h;  
      fprintf(fp, "%f,%f,%f,%f,%f \n",y[i][0],y[i][1],y[i][2],y[i][3],y[i][4]);
    }
  fclose(fp);
}
