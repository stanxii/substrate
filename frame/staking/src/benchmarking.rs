use super::*;
use sp_runtime::{BenchmarkParameter, BenchmarkResult};
use sp_runtime::traits::{Dispatchable, Convert, Benchmarking};
use sp_io::hashing::blake2_256;
use sp_std::convert::TryInto;
use frame_support::{StoragePrefixedMap, StorageValue};
use pallet_indices::address::Address;
use frame_system::RawOrigin;
use sp_phragmen::{ExtendedBalance, StakedAssignment, reduce};

macro_rules! assert_ok {
	( $x:expr $(,)? ) => {
		assert_eq!($x, Ok(()));
	};
	( $x:expr, $y:expr $(,)? ) => {
		assert_eq!($x, Ok($y));
	}
}

const CTRL_PREFIX: u32 = 1000;
const NOMINATOR_PREFIX: u32 = 1_000_000;
const USER: u32 = 999_999_999;
const ED: u32 = 1_000_000_000;

type AddressOf<T> = Address<<T as frame_system::Trait>::AccountId, u32>;


fn rr(a: u32, b: u32) -> u32 {
	use rand::Rng;
	use rand_chacha::rand_core::SeedableRng;
	// well, what do you expect?
	let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(999_666_777);
	rng.gen_range(a, b)
}

fn account<T: Trait>(index: u32) -> T::AccountId {
	let entropy = (b"benchmark/staking", index).using_encoded(blake2_256);
	T::AccountId::decode(&mut &entropy[..]).unwrap_or_default()
}

fn address<T: Trait>(index: u32) -> AddressOf<T> {
	pallet_indices::address::Address::Id(account::<T>(index))
}

fn signed<T: Trait>(who: T::AccountId) -> T::Origin {
	RawOrigin::Signed(who).into()
}

fn signed_account<T: Trait>(index: u32) -> T::Origin {
	signed::<T>(account::<T>(index))
}

fn bond_validator<T: Trait>(stash: T::AccountId, ctrl: u32, val: BalanceOf<T>)
	where T::Lookup: StaticLookup<Source=AddressOf<T>>
{
	let _ = T::Currency::make_free_balance_be(&stash, val);
	assert_ok!(<Module<T>>::bond(
		signed::<T>(stash),
		address::<T>(ctrl),
		val,
		RewardDestination::Controller
	));
	assert_ok!(<Module<T>>::validate(
		signed_account::<T>(ctrl),
		ValidatorPrefs::default()
	));
}

fn bond_nominator<T: Trait>(
	stash: T::AccountId,
	ctrl: u32,
	val: BalanceOf<T>,
	target: Vec<AddressOf<T>>
) where T::Lookup: StaticLookup<Source=AddressOf<T>> {

	let _ = T::Currency::make_free_balance_be(&stash, val);
	assert_ok!(<Module<T>>::bond(
		signed::<T>(stash),
		address::<T>(ctrl),
		val,
		RewardDestination::Controller
	));
	assert_ok!(<Module<T>>::nominate(signed_account::<T>(ctrl), target));
}

fn setup_with_no_solution_on_chain<T: Trait>(
	num_stakers: u32,
	num_voters: u32,
	edge_per_voter: u32,
) where T::Lookup: StaticLookup<Source=AddressOf<T>> {
	(0..num_stakers).for_each(|i| {
		bond_validator::<T>(
			account::<T>(i),
			i + CTRL_PREFIX,
			rr(ED, 2*ED).into(),
		)
	});

	(0..num_voters).for_each(|i| {
		let mut targets: Vec<AddressOf<T>> = Vec::with_capacity(edge_per_voter as usize);
		let mut all_targets = (0..num_stakers).map(|t| address::<T>(t)).collect::<Vec<_>>();
		assert!(num_stakers > edge_per_voter);
		(0..edge_per_voter).for_each(|_| {
			let target = all_targets.remove(rr(0, all_targets.len() as u32 - 1) as usize);
			targets.push(target);
		});
		bond_nominator::<T>(
			account::<T>(i + NOMINATOR_PREFIX),
			i + NOMINATOR_PREFIX + CTRL_PREFIX,
			rr(ED, 2*ED).into(),
			targets,
		)
	});
}

fn get_weak_solution<T: Trait>()
-> (Vec<T::AccountId>, CompactAssignments<T::AccountId, ExtendedBalance>) {
	use sp_std::collections::btree_map::BTreeMap;
	let mut backing_stake_of: BTreeMap<T::AccountId, BalanceOf<T>> = BTreeMap::new();

	// self stake
	<Validators<T>>::enumerate().for_each(|(who, _p)| {
		*backing_stake_of.entry(who.clone()).or_insert(Zero::zero()) +=
			<Module<T>>::slashable_balance_of(&who)
	});

	// add nominator stuff
	<Nominators<T>>::enumerate().for_each(|(who, nomination)| {
		nomination.targets.into_iter().for_each(|v| {
			*backing_stake_of.entry(v).or_insert(Zero::zero()) +=
				<Module<T>>::slashable_balance_of(&who)
		})
	});

	// elect winners
	let mut sorted: Vec<T::AccountId> = backing_stake_of.keys().cloned().collect();
	sorted.sort_by_key(|x| backing_stake_of.get(x).unwrap());
	let winners: Vec<T::AccountId> = sorted
		.iter()
		.cloned()
		.take(<Module<T>>::validator_count() as usize)
		.collect();

	let mut assignments: Vec<StakedAssignment<T::AccountId>> = Vec::new();
	<Nominators<T>>::enumerate().for_each(|(who, nomination)| {
		let mut dist: Vec<(T::AccountId, ExtendedBalance)> = Vec::new();
		nomination.targets.into_iter().for_each(|v| {
			if winners.iter().find(|&w| *w == v).is_some() {
				dist.push((v, ExtendedBalance::zero()));
			}
		});

		if dist.len() == 0 {
			return;
		}

		// assign real stakes. just split the stake.
		let stake = <T::CurrencyToVote as Convert<BalanceOf<T>, u64>>::convert(
			<Module<T>>::slashable_balance_of(&who),
		) as ExtendedBalance;
		let mut sum: ExtendedBalance = Zero::zero();
		let dist_len = dist.len();

		// assign main portion
		dist.iter_mut().for_each(|(_, w)| {
			let partial = stake / (dist_len as ExtendedBalance);
			*w = partial;
			sum += partial;
		});

		// assign the leftover to last.
		let leftover = stake - sum;
		let last = dist.last_mut().unwrap();
		last.1 += leftover;

		assignments.push(StakedAssignment {
			who,
			distribution: dist,
		});
	});

	// submit it
	let compact = <CompactAssignments<T::AccountId, ExtendedBalance>>::from_staked(assignments);

	(winners, compact)
}

fn get_seq_phragmen_solution<T: Trait>()
-> (Vec<T::AccountId>, CompactAssignments<T::AccountId, ExtendedBalance>) {
	let sp_phragmen::PhragmenResult {
		winners,
		assignments,
	} = <Module<T>>::do_phragmen().unwrap();
	let winners = winners.into_iter().map(|(w, _)| w).collect();

	let mut staked: Vec<StakedAssignment<T::AccountId>> = assignments
		.into_iter()
		.map(|a| a.into_staked::<_, _, T::CurrencyToVote>(<Module<T>>::slashable_balance_of))
		.collect();

	reduce(&mut staked);
	let compact = <CompactAssignments<T::AccountId, ExtendedBalance>>::from_staked(staked);

	(winners, compact)
}

fn clean<T: Trait>() {
	<Validators<T>>::enumerate().for_each(|(k, _)| <Validators<T>>::remove(k));
	<Nominators<T>>::enumerate().for_each(|(k, _)| <Nominators<T>>::remove(k));
	<Stakers<T>>::remove_all();
	<Ledger<T>>::remove_all();
	<Bonded<T>>::remove_all();
	<QueuedElected<T>>::kill();
}

#[repr(u32)]
#[allow(dead_code)]
#[derive(Debug)]
pub enum BenchmarkingMode {
	InitialSubmission,
	StrongerSubmission,
	WeakerSubmission,
}

impl<T: Trait> Benchmarking<BenchmarkParameter, BenchmarkResult> for Module<T> where
	T::Lookup: StaticLookup<Source=AddressOf<T>>,
{
	fn run_benchmark(parameters: Vec<(BenchmarkParameter, u32)>, repeat: u32) -> Vec<BenchmarkResult> {
		let mut results: Vec<BenchmarkResult> = Vec::new();

		let param = |x| parameters.iter().find(|&p| p.0 == x).map(|p| p.1).expect("Unexpected param");

		// Just set this once.
		<EraElectionStatus<T>>::put(ElectionStatus::Open(T::BlockNumber::from(1u32)));

		for _r in 0..repeat {
			sp_std::if_std! {
				println!("Repeat std {}", _r);
			}
			let num_stakers = param(BenchmarkParameter::S);
			let num_voters = param(BenchmarkParameter::V);
			let edge_per_voter = param(BenchmarkParameter::E);
			let mode: BenchmarkingMode = unsafe {
				sp_std::mem::transmute(param(BenchmarkParameter::M))
			};

			// stake and nominate everyone
			setup_with_no_solution_on_chain::<T>(num_stakers, num_voters, edge_per_voter);

			sp_std::if_std! {
				println!("state is set. Mode = {:?}", mode);
			}

			let (winners, compact) = match mode {
				BenchmarkingMode::InitialSubmission => {
					/* No need to setup anything */
					get_seq_phragmen_solution::<T>()
				},
				BenchmarkingMode::StrongerSubmission => {
					let (winners, compact) = get_weak_solution::<T>();
					assert_ok!(
						<Module<T>>::submit_election_solution(
							signed_account::<T>(USER),
							winners,
							compact
						)
					);
					get_seq_phragmen_solution::<T>()
				},
				BenchmarkingMode::WeakerSubmission => {
					let (winners, compact) = get_seq_phragmen_solution::<T>();
					assert_ok!(
						<Module<T>>::submit_election_solution(
							signed_account::<T>(USER),
							winners,
							compact
						)
					);
					get_weak_solution::<T>()
				}
			};

			sp_std::if_std! {
				println!("Rest is also done. Mode = {:?}", mode);
			}

			let call = crate::Call::<T>::submit_election_solution(
				winners,
				compact,
			);
			let start = sp_io::benchmarking::current_time();
			sp_std::if_std! {
				println!("Repeat std {}", _r);
			}
			sp_runtime::print("Repeat");
			sp_runtime::print(_r);
			assert_ok!(call.dispatch(signed_account::<T>(USER)));
			let finish = sp_io::benchmarking::current_time();
			results.push(finish - start);

			clean::<T>();
		}

		results
	}
}

#[cfg(test)]
mod tests {
	use crate::{Module};
	use crate::mock::*;
	use super::*;

	use sp_runtime::traits::Benchmarking;

	#[test]
	fn basic_setup_works() {
		let mut ext = sp_io::TestExternalities::new(Default::default());

		ext.execute_with(|| {
			let parameters = vec![
				(BenchmarkParameter::S, 10),
				(BenchmarkParameter::V, 10),
				(BenchmarkParameter::E, 10),
				(BenchmarkParameter::M, 0),
			];
			// TODO
		})
	}
}
