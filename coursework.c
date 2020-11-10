/**
 * The function to optimise as part of the coursework.
 *
 * l0, l1, l2 and l3 record the amount of time spent in each loop
 * and should not be optimised out. :)
 */
#include <xmmintrin.h>
#include <emmintrin.h>

void compute() {

	double t0, t1;

	// Loop 0.
	t0 = wtime();
	int size = (N/8)*8;
	int i;
	const __m256 value = _mm256_set1_ps(0.0f);
	for (i = 0; i < size; i+=8) {
		_mm256_store_ps(&ax[i], value);
	}
	for(;i<N;i++)
	{
		ax[i] = 0.0f;
	}
	for (i = 0; i < size; i+=8)
	{
	   _mm256_store_ps(&ay[i], value);
	}
	for(;i<N;i++)
	{
		ay[i] = 0.0f;
	}
	for (i = 0; i < size; i+=8)
	{
	   _mm256_store_ps(&az[i], value);
	}
	for(;i<N;i++)
	{
		az[i] = 0.0f;
	}

	t1 = wtime();
	l0 += (t1 - t0);

    // Loop 1.
    t0 = wtime();
    __m256 eps1 = _mm256_set1_ps(eps);
  	#pragma omp parallel for
  	for (int i = 0; i < size; i+=8)
    {
  		for (int j = 0; j < N; j++)
       {
        __m256 xi = _mm256_load_ps(x+i);
        __m256 yi = _mm256_load_ps(y+i);
        __m256 zi = _mm256_load_ps(z+i);
        __m256 mj =  _mm256_set1_ps(m[j]);
  			__m256 rx = _mm256_sub_ps(_mm256_set1_ps(x[j]),xi);
  			__m256 ry = _mm256_sub_ps(_mm256_set1_ps(y[j]),yi);
  			__m256 rz = _mm256_sub_ps(_mm256_set1_ps(z[j]),zi);
  			__m256 sum1 = _mm256_add_ps(_mm256_mul_ps(rx,rx),_mm256_mul_ps(ry,ry));
  			__m256 sum2 = _mm256_add_ps(_mm256_mul_ps(rz,rz),eps1);
  			__m256 r2 = _mm256_add_ps(sum1,sum2);
  			__m256 r2inv = _mm256_rsqrt_ps(r2);
  			__m256 r6inv= _mm256_mul_ps(_mm256_mul_ps(r2inv,r2inv),r2inv);
  			__m256 s = _mm256_mul_ps(mj,r6inv);
  			__m256 ax1 = _mm256_add_ps(_mm256_load_ps(ax+i),_mm256_mul_ps(s,rx));
  			__m256 ay1 = _mm256_add_ps(_mm256_load_ps(ay+i),_mm256_mul_ps(s,ry));
  			__m256 az1 = _mm256_add_ps(_mm256_load_ps(az+i),_mm256_mul_ps(s,rz));
  			_mm256_store_ps(ax+i,ax1);
  			_mm256_store_ps(ay+i,ay1);
  			_mm256_store_ps(az+i,az1);
  		}
  	}
  	for (; i < N; i++)
     {
  		for (int j=0; j<N; j++)
      {
  			float rx = x[j] - x[i];
  			float ry = y[j] - y[i];
  			float rz = z[j] - z[i];
  			float r2 = rx*rx + ry*ry + rz*rz + eps;
  			float r2inv = 1.0f / sqrt(r2);
  			float r6inv = r2inv * r2inv * r2inv;
  			float s = m[j] * r6inv;
  			ax[i] += s * rx;
  			ay[i] += s * ry;
  			az[i] += s * rz;
  		}
  	}

    t1 = wtime();
    l1 += (t1 - t0);


	// Loop 2.
	t0 = wtime();
	 __m256 xdmp = _mm256_set1_ps(dmp);
	 __m256 xdt = _mm256_set1_ps(dt);
	float avx[8],avy[8],avz[8];
	for (int i = 0; i < size; i+=8)
   {

  	__m256 temp0 = _mm256_mul_ps(xdt , _mm256_load_ps (ax+i));
  	__m256 xvx = _mm256_mul_ps(xdmp,temp0);
  	_mm256_store_ps(avx,xvx);
    __m256 vx_v  =_mm256_add_ps(_mm256_load_ps(vx+i),xvx);
    _mm256_store_ps(vx+i, vx_v);

  }
  for(; i < N ;i++)
  {
    vx[i] += dmp * (dt * ax[i]);
  }
  for (int i = 0; i < size; i+=8)
   {
	   __m256 temp1 = _mm256_mul_ps(xdt , _mm256_load_ps (ay+i));
	   __m256 xvy = _mm256_mul_ps(xdmp,temp1);
	   _mm256_store_ps(avy,xvy);
     __m256 vy_v  =_mm256_add_ps(_mm256_load_ps(vy+i),xvy);
     _mm256_store_ps(vy+i, vy_v);
   }
  for(; i < N ;i++)
  {
    vy[i] += dmp * (dt * ay[i]);
  }
  for (int i = 0; i < size; i+=8)
  {
	   __m256 temp2 = _mm256_mul_ps(xdt , _mm256_load_ps (az+i));
	   __m256 xvz =_mm256_mul_ps(xdmp,temp2);
	   _mm256_store_ps(avz,xvz);
     __m256 vz_v  =_mm256_add_ps(_mm256_load_ps(vz+i),xvz);
     _mm256_store_ps(vz+i, vz_v);
  }
   for(; i < N ;i++)
   {
     vz[i] += dmp * (dt * az[i]);
   }

	t1 = wtime();
	l2 += (t1 - t0);

	// Loop 3.
	t0 = wtime();
	__m256 upbound = _mm256_set1_ps(1.0f);
	__m256 lowbound = _mm256_set1_ps(-1.0f);
	for (int i = 0; i < size; i+=8)
  {

  	__m256 v_x = _mm256_load_ps(vx+i);
    __m256 xi  = _mm256_load_ps(x+i);

  	__m256 sum  = _mm256_add_ps(xi, _mm256_mul_ps(xdt, v_x));
  	_mm256_store_ps(x+i,sum);
  	__m256 compare = _mm256_or_ps(_mm256_cmp_ps (sum,lowbound,_CMP_LE_OQ),_mm256_cmp_ps (sum,upbound,_CMP_GE_OQ));
  	__m256 minus = _mm256_mul_ps(v_x,lowbound);
  	__m256 result = _mm256_or_ps(_mm256_and_ps(compare,minus),_mm256_andnot_ps(compare,v_x));
  	_mm256_store_ps(vx+i,result);
	}
	for(;i<N;i++)
	{
		x[i] += dt * vx[i];
			if (x[i] >= 1.0f || x[i] <= -1.0f) vx[i] *= -1.0f;
	}

	for (int i = 0; i < size; i+=8)
   {

  	__m256 v_y = _mm256_load_ps(vy+i);
    __m256 yi  = _mm256_load_ps(y+i);

  	__m256 sum  = _mm256_add_ps(yi, _mm256_mul_ps(xdt, v_y));
  	_mm256_store_ps(y+i,sum);
  	__m256 compare = _mm256_or_ps(_mm256_cmp_ps (sum,lowbound,_CMP_LE_OQ),_mm256_cmp_ps (sum,upbound,_CMP_GE_OQ));
  	__m256 minus = _mm256_mul_ps(v_y,lowbound);
  	__m256 result = _mm256_or_ps(_mm256_and_ps(compare,minus),_mm256_andnot_ps(compare,v_y));
  	_mm256_store_ps(vy+i,result);
	}
	for(;i<N;i++)
	{
		y[i] += dt * vy[i];
			if (y[i] >= 1.0f || y[i] <= -1.0f) vy[i] *= -1.0f;
	}
	for (int i = 0; i < size; i+=8)
  {

  	__m256 v_z = _mm256_load_ps(vz+i);
    __m256 zi  = _mm256_load_ps(z+i);

  	__m256 sum  = _mm256_add_ps(zi, _mm256_mul_ps(xdt, v_z));
  	_mm256_store_ps(z+i,sum);
    __m256 compare = _mm256_or_ps(_mm256_cmp_ps (sum,lowbound,_CMP_LE_OQ),_mm256_cmp_ps (sum,upbound,_CMP_GE_OQ));
  	__m256 minus = _mm256_mul_ps(v_z,lowbound);
  	__m256 result = _mm256_or_ps(_mm256_and_ps(compare,minus),_mm256_andnot_ps(compare,v_z));
  	_mm256_store_ps(vz+i,result);
	}

	for(;i<N;i++)
	{
		z[i] += dt * vz[i];
			if (z[i] >= 1.0f || z[i] <= -1.0f) vz[i] *= -1.0f;
	}

	t1 = wtime();
	l3 += (t1 - t0);

}
