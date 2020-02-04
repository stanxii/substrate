

#![cfg_attr(not(feature = "std"), no_std)]

sp_api::decl_runtime_apis! {
	pub trait StakingBenchmark<Parameter: codec::Codec, Res: codec::Codec> {
		fn run_benchmark(
			parameters: Parameter,
			repeat: u32
		) -> Res;
	}
}
