#include <benchmark/benchmark.h>
#include <array>
#include <cytnx.hpp>
#include <itensor/all.h>
 
// Cytnx test
static void BM_Cytnx_declare(benchmark::State& state)
{
	for (auto _: state) 
	{
		cytnx::Tensor A;
	}
}
BENCHMARK(BM_Cytnx_declare);

static void BM_Cytnx_contract(benchmark::State& state)
{
	for (auto _: state) {
		auto A = cytnx::UniTensor(cytnx::ones({3, 3, 3}));
		A.set_labels(std::vector<long int>{1l, 2l, 3l});
		auto B = cytnx::UniTensor(cytnx::ones({3, 3, 3, 3}));
		B.set_labels(std::vector<long int>{2l, 3l, 4l, 5l});
		auto C = cytnx::Contract(A, B);
	}
}
BENCHMARK(BM_Cytnx_contract);

//test with several arguments, ex, bond dimension
static void TestBenchmark(benchmark::State& state) 
{
	int i = state.range(0);
	int j = state.range(1);
	for (auto _ : state) 
	{
		auto A = cytnx::UniTensor(cytnx::ones({i, i, 3}));
	}
}
BENCHMARK(TestBenchmark)
	->Args({5, 3})
	->Args({10, 9});
;

// itensor test
static void BM_itensor_declare(benchmark::State& state)
{
	for (auto _: state) 
	{
		auto i = itensor::Index(4, "index i");
		auto j = itensor::Index(6, "index j");
		auto T = itensor::ITensor(i,j);
		T.set(i=3, j = 2, 3.14159);
	}
}
BENCHMARK(BM_itensor_declare);
 
BENCHMARK_MAIN();
