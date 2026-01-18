/*
 * ============================================================================
 *
 *        Authors:  Prashant Pandey <ppandey@cs.stonybrook.edu>
 *                  Rob Johnson <robj@vmware.com>
 *
 * ============================================================================
 */

#ifndef _GQF_INT_CUH_
#define _GQF_INT_CUH_

#include <inttypes.h>
#include <stdbool.h>

#include "gqf.cuh"
#include "partitioned_counter.cuh"

#ifdef __cplusplus
extern "C" {
#endif

#define MAGIC_NUMBER 1018874902021329732

/* Can be
   0 (choose size at run-time),
   8, 16, 32, or 64 (for optimized versions),
   or other integer <= 56 (for compile-time-optimized bit-shifting-based versions)
*/

#ifndef QF_BITS_PER_SLOT
    #define QF_BITS_PER_SLOT 32
#endif

#ifndef R_BITS
    #define R_BITS 16
#endif

/* Must be >= 6.  6 seems fastest. */
#define QF_BLOCK_OFFSET_BITS (6)

#define QF_SLOTS_PER_BLOCK (1ULL << QF_BLOCK_OFFSET_BITS)
#define QF_METADATA_WORDS_PER_BLOCK ((QF_SLOTS_PER_BLOCK + 63) / 64)

	typedef struct{
		/* Code works with uint16_t, uint32_t, etc, but uint8_t seems just as fast as
		 * anything else */
		uint8_t offset;
		uint64_t occupieds[QF_METADATA_WORDS_PER_BLOCK];
		uint64_t runends[QF_METADATA_WORDS_PER_BLOCK];
#if QF_BITS_PER_SLOT == 8
		uint8_t  slots[QF_SLOTS_PER_BLOCK];
#elif QF_BITS_PER_SLOT == 16
		uint16_t  slots[QF_SLOTS_PER_BLOCK];
#elif QF_BITS_PER_SLOT == 32
		uint32_t  slots[QF_SLOTS_PER_BLOCK];
#elif QF_BITS_PER_SLOT == 64
		uint64_t  slots[QF_SLOTS_PER_BLOCK];
#elif QF_BITS_PER_SLOT != 0
		uint8_t   slots[QF_SLOTS_PER_BLOCK * QF_BITS_PER_SLOT / 8];
#else
		uint8_t   slots[];
#endif
	}  __attribute__((__packed__)) qfblock;

//	struct __attribute__ ((__packed__)) qfblock;
//	typedef struct qfblock qfblock;

  typedef struct file_info {
		int fd;
		char *filepath;
	} file_info;

	// The below struct is used to instrument the code.
	// It is not used in normal operations of the CQF.
	typedef struct {
		uint64_t total_time_single;
		uint64_t total_time_spinning;
		uint64_t locks_taken;
		uint64_t locks_acquired_single_attempt;
	} wait_time_data;

	typedef struct quotient_filter_runtime_data {
		file_info f_info;
		uint32_t auto_resize;
		int64_t (*container_resize)(QF *qf, uint64_t nslots);
		pc_t pc_nelts;
		pc_t pc_ndistinct_elts;
		pc_t pc_noccupied_slots;
		uint64_t num_locks;
		volatile int metadata_lock;
		uint16_t *locks;

		//extra additions needed to move buffers into main code
		uint64_t ** buffers;
		uint64_t * buffer_sizes;

		wait_time_data *wait_times;
	} quotient_filter_runtime_data;

	typedef quotient_filter_runtime_data qfruntime;

	typedef struct quotient_filter_metadata {
		uint64_t magic_endian_number;
		enum qf_hashmode hash_mode;
		uint32_t reserved;
		uint64_t total_size_in_bytes;
		uint32_t seed;
		uint64_t nslots;
		uint64_t xnslots;
		uint64_t key_bits;
		uint64_t value_bits;
		uint64_t key_remainder_bits;
		uint64_t bits_per_slot;
		uint64_t range;
		uint64_t nblocks;
		uint64_t nelts;
		uint64_t ndistinct_elts;
		uint64_t noccupied_slots;
		bool qf_full;
	} quotient_filter_metadata;

	typedef quotient_filter_metadata qfmetadata;

	typedef struct quotient_filter {
		qfruntime *runtimedata;
		qfmetadata *metadata;
		qfblock *blocks;
	} quotient_filter;

	typedef quotient_filter QF;


	// The below struct is used to instrument the code.
	// It is not used in normal operations of the CQF.
	typedef struct cluster_data {
		uint64_t start_index;
		uint16_t length;
	} cluster_data;
	typedef struct quotient_filter_iterator {
		const QF *qf;
		uint64_t run;
		uint64_t current;
		uint64_t cur_start_index;
		uint16_t cur_length;
		uint32_t num_clusters;
		cluster_data *c_info;
	} quotient_filter_iterator;

#ifdef __cplusplus
}
#endif


#endif /* _GQF_INT_CUH_ */
