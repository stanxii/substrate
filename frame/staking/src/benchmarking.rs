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

static mut SEED: u64 = 999_666;

type AddressOf<T> = Address<<T as frame_system::Trait>::AccountId, u32>;

fn rr(a: u32, b: u32) -> u32 {
	use rand::Rng;
	use rand_chacha::rand_core::SeedableRng;
	// well, what do you expect?
	unsafe {
		let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(SEED);
		SEED += 1;
		rng.gen_range(a, b)
	}
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
		);
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
		);
	});


}

fn get_weak_solution<T: Trait>(do_reduce: bool)
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
		let dist_len = dist.len() as ExtendedBalance;

		// assign main portion
		// only take the first half into account. This should highly imbalance stuff?
		dist
			.iter_mut()
			.take( if dist_len > 1 { (dist_len as usize) / 2 } else { 1 } )
			.for_each(|(_, w)|
		{
			let partial = stake / dist_len;
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

	// add self support to winners.
	winners.iter().for_each(|w| assignments.push(StakedAssignment {
		who: w.clone(),
		distribution: vec![(
			w.clone(),
			<T::CurrencyToVote as Convert<BalanceOf<T>, u64>>::convert(
				<Module<T>>::slashable_balance_of(&w)
			) as ExtendedBalance,
		)]
	}));

	let support = build_support_map::<T::AccountId>(&winners, &assignments).0;
	let score = evaluate_support(&support);

	if do_reduce {
		reduce(&mut assignments);
	}

	let compact = <CompactAssignments<T::AccountId, ExtendedBalance>>::from_staked(assignments);
	(winners, compact, score)
}

fn get_seq_phragmen_solution<T: Trait>(do_reduce: bool)
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

	if do_reduce {
		reduce(&mut staked);
	}

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

		for r in 0..repeat {
			// TODO: These are the parameters of the benchmark.
			let num_validators = 300;
			let num_voters = 600;
			let edge_per_voter = 12;
			let mode: BenchmarkingMode = BenchmarkingMode::StrongerSubmission;
			let do_reduce: bool = true;
			// select all of them
			let to_elect: u32 = 10;


			ValidatorCount::put(to_elect);
			MinimumValidatorCount::put(to_elect/2);
			<EraElectionStatus<T>>::put(ElectionStatus::Open(T::BlockNumber::from(1u32)));

			// stake and nominate everyone
			setup_with_no_solution_on_chain::<T>(num_validators, num_voters, edge_per_voter);

			let (winners, compact, score) = match mode {
				BenchmarkingMode::InitialSubmission => {
					/* No need to setup anything */
					get_seq_phragmen_solution::<T>(do_reduce)
				},
				BenchmarkingMode::StrongerSubmission => {
					let (winners, compact, score) = get_weak_solution::<T>(false);
					sp_std::if_std! {
						println!("Weak solution submitting with {:?}", score);
					}
					assert_ok!(
						<Module<T>>::submit_election_solution(
							signed_account::<T>(USER),
							winners,
							compact,
							score,
						)
					);
					get_seq_phragmen_solution::<T>(do_reduce)
				},
				BenchmarkingMode::WeakerSubmission => {
					let (winners, compact, score) = get_seq_phragmen_solution::<T>(do_reduce);
					assert_ok!(
						<Module<T>>::submit_election_solution(
							signed_account::<T>(USER),
							winners,
							compact,
							score,
						)
					);
					get_weak_solution::<T>(do_reduce)
				}
			};

			sp_std::if_std! {
				println!("Setup is done. Mode = {:?} Submitting {:?} iter {}/{}", mode, score, r, repeat);
			}

			assert_eq!(winners.len() as u32, <Module<T>>::validator_count());

			let call = crate::Call::<T>::submit_election_solution(
				winners,
				compact,
				score,
			);

			#[cfg(not(test))]
			sp_io::benchmarking::commit_db();
			let start = sp_io::benchmarking::current_time();
			assert_ok!(call.dispatch(signed_account::<T>(USER)));
			let finish = sp_io::benchmarking::current_time();
			let elapsed = finish - start;

			#[cfg(test)]
			clean::<T>();
			#[cfg(not(test))]
			sp_io::benchmarking::wipe_db();

			results.push((Default::default(), elapsed));
		}
		results
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use sp_runtime::traits::Benchmarking;
	use sp_runtime::BenchmarkResults;
	use frame_support::{impl_outer_origin, impl_outer_dispatch, parameter_types};

	type AccountId = u64;
	type AccountIndex = u32;
	type BlockNumber = u64;
	type Balance = u64;

	type Balances = pallet_balances::Module<Test>;
	type Staking = crate::Module<Test>;
	type Indices = pallet_indices::Module<Test>;

	impl_outer_origin! {
		pub enum Origin for Test  where system = frame_system {}
	}

	impl_outer_dispatch! {
		pub enum Call for Test where origin: Origin {
			staking::Staking,
		}
	}

	#[derive(Clone, Eq, PartialEq, Debug)]
	pub struct Test;

	impl frame_system::Trait for Test {
		type Origin = Origin;
		type Index = AccountIndex;
		type BlockNumber = BlockNumber;
		type Call = Call;
		type Hash = sp_core::H256;
		type Hashing = ::sp_runtime::traits::BlakeTwo256;
		type AccountId = AccountId;
		type Lookup = Indices;
		type Header = sp_runtime::testing::Header;
		type Event = ();
		type BlockHashCount = ();
		type MaximumBlockWeight = ();
		type AvailableBlockRatio = ();
		type MaximumBlockLength = ();
		type Version = ();
		type ModuleToIndex = ();
	}
	parameter_types! {
		pub const ExistentialDeposit: Balance = 0;
	}
	impl pallet_balances::Trait for Test {
		type Balance = Balance;
		type OnReapAccount = ();
		type OnNewAccount = ();
		type Event = ();
		type TransferPayment = ();
		type DustRemoval = ();
		type ExistentialDeposit = ExistentialDeposit;
		type CreationFee = ();
	}
	impl pallet_indices::Trait for Test {
		type AccountIndex = AccountIndex;
		type IsDeadAccount = Balances;
		type ResolveHint = pallet_indices::SimpleResolveHint<Self::AccountId, Self::AccountIndex>;
		type Event = ();
	}
	parameter_types! {
		pub const MinimumPeriod: u64 = 5;
	}
	impl pallet_timestamp::Trait for Test {
		type Moment = u64;
		type OnTimestampSet = ();
		type MinimumPeriod = MinimumPeriod;
	}
	impl pallet_session::historical::Trait for Test {
		type FullIdentification = crate::Exposure<AccountId, Balance>;
		type FullIdentificationOf = crate::ExposureOf<Test>;
	}

	sp_runtime::impl_opaque_keys! {
		pub struct SessionKeys {
			pub foo: sp_runtime::testing::UintAuthorityId,
		}
	}

	pub struct TestSessionHandler;
	impl pallet_session::SessionHandler<AccountId> for TestSessionHandler {
		// EVEN if no tests break, I must have broken something here... TODO
		const KEY_TYPE_IDS: &'static [sp_runtime::KeyTypeId] = &[];

		fn on_genesis_session<Ks: sp_runtime::traits::OpaqueKeys>(_validators: &[(AccountId, Ks)]) {}

		fn on_new_session<Ks: sp_runtime::traits::OpaqueKeys>(
			_: bool,
			_: &[(AccountId, Ks)],
			_: &[(AccountId, Ks)],
		) {}

		fn on_disabled(_: usize) {}
	}

	impl pallet_session::Trait for Test {
		type SessionManager = pallet_session::historical::NoteHistoricalRoot<Test, Staking>;
		type Keys = SessionKeys;
		type ShouldEndSession = pallet_session::PeriodicSessions<(), ()>;
		type SessionHandler = TestSessionHandler;
		type Event = ();
		type ValidatorId = AccountId;
		type ValidatorIdOf = crate::StashOf<Test>;
		type DisabledValidatorsThreshold = ();
	}
	pallet_staking_reward_curve::build! {
		const I_NPOS: sp_runtime::curve::PiecewiseLinear<'static> = curve!(
			min_inflation: 0_025_000,
			max_inflation: 0_100_000,
			ideal_stake: 0_500_000,
			falloff: 0_050_000,
			max_piece_count: 40,
			test_precision: 0_005_000,
		);
	}
	parameter_types! {
		pub const RewardCurve: &'static sp_runtime::curve::PiecewiseLinear<'static> = &I_NPOS;
	}

	pub type Extrinsic = sp_runtime::testing::TestXt<Call, ()>;
	type SubmitTransaction = frame_system::offchain::TransactionSubmitter<sp_runtime::testing::UintAuthorityId, Test, Extrinsic>;

	impl crate::Trait for Test {
		type Currency = Balances;
		type Time = pallet_timestamp::Module<Self>;
		type CurrencyToVote = mock::CurrencyToVoteHandler;
		type RewardRemainder = ();
		type Event = ();
		type Slash = ();
		type Reward = ();
		type SessionsPerEra = ();
		type SlashDeferDuration = ();
		type SlashCancelOrigin = frame_system::EnsureRoot<Self::AccountId>;
		type BondingDuration = ();
		type SessionInterface = Self;
		type RewardCurve = RewardCurve;
		type NextSessionChange = mock::PeriodicSessionChange<()>;
		type ElectionLookahead = ();
		type Call = Call;
		type SubmitTransaction = SubmitTransaction;
		type KeyType = mock::dummy_sr25519::AuthorityId;
	}


	fn new_test_ext() -> sp_io::TestExternalities {
		frame_system::GenesisConfig::default().build_storage::<Test>().unwrap().into()
	}

	#[test]
	fn it_simply_should_work() {
		new_test_ext().execute_with(|| {
			<Staking as Benchmarking<BenchmarkResults>>::run_benchmark(Default::default(), Default::default(), 1);
		})
	}
}
