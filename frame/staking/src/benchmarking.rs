use super::*;
use sp_runtime::{BenchmarkResults};
use sp_runtime::traits::{Dispatchable, Convert, Benchmarking};
use sp_io::hashing::blake2_256;
use frame_support::{StoragePrefixedMap, StorageValue};
use pallet_indices::address::Address;
use frame_system::RawOrigin;
use sp_phragmen::{
	ExtendedBalance, StakedAssignment, reduce, build_support_map, evaluate_support, PhragmenScore,
};

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
-> (Vec<T::AccountId>, CompactAssignments<T::AccountId, ExtendedBalance>, PhragmenScore) {
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

	let support = build_support_map::<T::AccountId>(&winners, &assignments).0;
	let score = evaluate_support(&support);

	let compact = <CompactAssignments<T::AccountId, ExtendedBalance>>::from_staked(assignments);
	(winners, compact, score)
}

fn get_seq_phragmen_solution<T: Trait>()
-> (Vec<T::AccountId>, CompactAssignments<T::AccountId, ExtendedBalance>, PhragmenScore) {
	let sp_phragmen::PhragmenResult {
		winners,
		assignments,
	} = <Module<T>>::do_phragmen().unwrap();
	let winners = winners.into_iter().map(|(w, _)| w).collect();

	let mut staked: Vec<StakedAssignment<T::AccountId>> = assignments
		.into_iter()
		.map(|a| a.into_staked::<_, _, T::CurrencyToVote>(<Module<T>>::slashable_balance_of, true))
		.collect();

	reduce(&mut staked);
	let (support_map, _) = build_support_map::<T::AccountId>(&winners, &staked);
	let score = evaluate_support::<T::AccountId>(&support_map);

	let compact = <CompactAssignments<T::AccountId, ExtendedBalance>>::from_staked(staked);
	(winners, compact, score)
}

fn clean<T: Trait>() {
	<Validators<T>>::enumerate().for_each(|(k, _)| <Validators<T>>::remove(k));
	<Nominators<T>>::enumerate().for_each(|(k, _)| <Nominators<T>>::remove(k));
	<Stakers<T>>::remove_all();
	<Ledger<T>>::remove_all();
	<Bonded<T>>::remove_all();
	<QueuedElected<T>>::kill();
	QueuedScore::kill();
}

#[repr(u32)]
#[allow(dead_code)]
#[derive(Debug)]
pub enum BenchmarkingMode {
	/// Initial submission. This will be rather cheap
	InitialSubmission,
	/// A better submission that will replace the previous ones. This is the most expensive.
	StrongerSubmission,
	/// A weak submission that will be rejected. This will be rather cheap.
	WeakerSubmission,
}

impl<T: Trait> Benchmarking<BenchmarkResults> for Module<T> where T::Lookup: StaticLookup<Source=AddressOf<T>> {
	fn run_benchmark(_extrinsic: Vec<u8>, _steps: u32, repeat: u32) -> Vec<BenchmarkResults> {
		let mut results: Vec<BenchmarkResults> = Vec::new();

		// Warm up the DB?
		sp_io::benchmarking::commit_db();
		sp_io::benchmarking::wipe_db();

		for r in 0..repeat {

			// Just set this once.
			<EraElectionStatus<T>>::put(ElectionStatus::Open(T::BlockNumber::from(1u32)));
			frame_support::storage::unhashed::put_raw(
				sp_core::storage::well_known_keys::HEAP_PAGES,
				&1_000_000_000_000_000u64.encode(),
			);

			// TODO: randomly generate these.
			let num_stakers = 300;
			let num_voters = 600;
			let edge_per_voter = 12;
			let mode: BenchmarkingMode = BenchmarkingMode::StrongerSubmission;

			// clean the state.
			clean::<T>();

			// stake and nominate everyone
			setup_with_no_solution_on_chain::<T>(num_stakers, num_voters, edge_per_voter);

			let (winners, compact, score) = match mode {
				BenchmarkingMode::InitialSubmission => {
					/* No need to setup anything */
					get_seq_phragmen_solution::<T>()
				},
				BenchmarkingMode::StrongerSubmission => {
					let (winners, compact, score) = get_weak_solution::<T>();
					assert_ok!(
						<Module<T>>::submit_election_solution(
							signed_account::<T>(USER),
							winners,
							compact,
							score,
						)
					);
					get_seq_phragmen_solution::<T>()
				},
				BenchmarkingMode::WeakerSubmission => {
					let (winners, compact, score) = get_seq_phragmen_solution::<T>();
					assert_ok!(
						<Module<T>>::submit_election_solution(
							signed_account::<T>(USER),
							winners,
							compact,
							score,
						)
					);
					get_weak_solution::<T>()
				}
			};

			sp_std::if_std! {
				println!("Setup is done. Mode = {:?} iter {}/{}", mode, r, repeat);
			}

			let call = crate::Call::<T>::submit_election_solution(
				winners,
				compact,
				score,
			);

			sp_io::benchmarking::commit_db();
			let start = sp_io::benchmarking::current_time();
			assert_ok!(call.dispatch(signed_account::<T>(USER)));
			let finish = sp_io::benchmarking::current_time();
			sp_io::benchmarking::wipe_db();

			results.push((Default::default(), finish - start));
		}
		results
	}
}
