#include<stdio.h>
#include<cuda.h>
__global__ void vecadd(int *a, int *b, int *c, int N){
	int tid=blockIdx.x*blockDim.x+threadIdx.x;
	 if(tid <N) c[tid]=a[tid]+b[tid];
}

int main(int argc, char *argv[]){
	if(argc!=3){
		printf("usage <executtable_name> <size_of_input> <seed>\n");
		exit(0);
         }
	int N=atoi(argv[1]);
	int seed=atoi(argv[2]);
	srand(seed);
   int *h_a,*h_b,*h_c;
   int *d_a,*d_b,*d_c;
   h_a=(int *)malloc(sizeof(int)*N);
   h_b=(int *)malloc(sizeof(int)*N);
   h_c=(int *)malloc(sizeof(int)*N);
  for(int i=0;i<N;i++){
   h_a[i]=rand()%1024;
   h_b[i]=rand()%1024;
   h_c[i]=0;
  }
cudaMalloc(&d_a,sizeof(int)*N);
cudaMalloc(&d_b,sizeof(int)*N);
cudaMalloc(&d_c,sizeof(int)*N);
if(cudaMemcpy(d_a,h_a,sizeof(int)*N,cudaMemcpyHostToDevice)!=cudaSuccess)printf("memcpy error");
if(cudaMemcpy(d_b,h_b,sizeof(int)*N,cudaMemcpyHostToDevice)!=cudaSuccess)printf("memcpy error");
if(cudaMemcpy(d_c,h_c,sizeof(int)*N,cudaMemcpyHostToDevice)!=cudaSuccess)printf("memcpy error");

//Launch the Kernel on Device
vecadd<<< N/1024+1, 1024>>>(d_a,d_b,d_c,N);
cudaDeviceSynchronize();
if(cudaMemcpy(h_c,d_c,sizeof(int)*N,cudaMemcpyDeviceToHost)!=cudaSuccess)printf("memcpy error");
for(int i=0;i<N;i++){
	int temp=h_a[i]+h_b[i];
	if(temp!=h_c[i]){printf("error in computation"); break;}
	if(i <10) printf(" h_c[%d]=%d , h_a[%d]=%d  h_b[%d]=%d\n", i, h_c[i], i, h_a[i],i,h_b[i]);
}
free(h_a);
free(h_b);
free(h_c);
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);


}
