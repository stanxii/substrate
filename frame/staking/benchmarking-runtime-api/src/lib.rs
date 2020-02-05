

#![cfg_attr(not(feature = "std"), no_std)]

sp_api::decl_runtime_apis! {
	pub trait StakingBenchmark<Ret: codec::Codec> {
		fn run_benchmarks() -> Ret;
	}
}
