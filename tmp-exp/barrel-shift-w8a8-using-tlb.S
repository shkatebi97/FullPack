	.arch armv8.2-a+crc+fp16
	.file	"barrel-shift-w8a8-using-tlb.cc"
	.text
	.section	.text._ZNKSt5ctypeIcE8do_widenEc,"axG",@progbits,_ZNKSt5ctypeIcE8do_widenEc,comdat
	.align	2
	.p2align 4,,11
	.weak	_ZNKSt5ctypeIcE8do_widenEc
	.type	_ZNKSt5ctypeIcE8do_widenEc, %function
_ZNKSt5ctypeIcE8do_widenEc:
.LFB5903:
	.cfi_startproc
	mov	w0, w1
	ret
	.cfi_endproc
.LFE5903:
	.size	_ZNKSt5ctypeIcE8do_widenEc, .-_ZNKSt5ctypeIcE8do_widenEc
	.text
	.align	2
	.p2align 4,,11
	.type	_Z7stepTLBPsPimmPa.constprop.0, %function
_Z7stepTLBPsPimmPa.constprop.0:
.LFB6665:
	.cfi_startproc
	stp	d8, d9, [sp, -384]!
	.cfi_def_cfa_offset 384
	.cfi_offset 72, -384
	.cfi_offset 73, -376
	mov	x2, x1
	stp	d10, d11, [sp, 16]
	stp	d12, d13, [sp, 32]
	stp	d14, d15, [sp, 48]
	.cfi_offset 74, -368
	.cfi_offset 75, -360
	.cfi_offset 76, -352
	.cfi_offset 77, -344
	.cfi_offset 78, -336
	.cfi_offset 79, -328
	ldp	q3, q2, [x0]
	ldp	q7, q17, [x0, 32]
	ldp	q6, q4, [x0, 64]
	ldp	q16, q5, [x0, 96]
#APP
// 23 "barrel-shift-w8a8-using-tlb.cc" 1
	mov w1, 0xFFFF
	mov w0, 0x0100
	ins v18.h[0], w0
	ins v18.h[1], w1
	mov w0, 0x1110
	ins v18.h[2], w0
	ins v18.h[3], w1
	mov w0, 0x2120
	ins v18.h[4], w0
	ins v18.h[5], w1
	mov w0, 0x3130
	ins v18.h[6], w0
	ins v18.h[7], w1
	mov w0, 0x02
	dup v1.16b, w0
	mov w1, 0x0000
	ins v1.h[1], w1
	ins v1.h[3], w1
	ins v1.h[5], w1
	ins v1.h[7], w1
	
// 0 "" 2
#NO_APP
	mov	v20.16b, v17.16b
	adrp	x0, .LC0
	mov	v8.16b, v6.16b
	add	x0, x0, :lo12:.LC0
	mov	v9.16b, v4.16b
	add	v0.16b, v18.16b, v1.16b
	ld1	{v12.16b - v15.16b}, [x0]
	adrp	x0, .LC0
	mov	v12.16b, v3.16b
	add	x0, x0, :lo12:.LC0
	mov	v28.16b, v5.16b
	mov	v21.16b, v13.16b
	mov	v22.16b, v14.16b
	mov	v23.16b, v15.16b
	ld1	{v24.16b - v27.16b}, [x0]
	add	x0, sp, 128
	mov	v10.16b, v14.16b
	mov	v11.16b, v15.16b
	mov	v24.16b, v16.16b
	st1	{v20.16b - v23.16b}, [x0]
	adrp	x0, .LC0
	add	x0, x0, :lo12:.LC0
	str	q6, [sp, 144]
	mov	v25.16b, v5.16b
	ld1	{v20.16b - v23.16b}, [x0]
	add	x0, sp, 64
	mov	v20.16b, v7.16b
	mov	v21.16b, v17.16b
	mov	v22.16b, v6.16b
	st1	{v8.16b - v11.16b}, [x0]
	add	x0, sp, 128
	mov	v13.16b, v2.16b
	stp	q16, q5, [sp, 96]
	ld1	{v8.16b - v11.16b}, [x0]
	add	x0, sp, 192
	mov	v10.16b, v4.16b
	mov	v14.16b, v7.16b
	mov	v15.16b, v17.16b
	st1	{v24.16b - v27.16b}, [x0]
	adrp	x0, .LC0
	add	x0, x0, :lo12:.LC0
	str	q3, [sp, 224]
	mov	v29.16b, v3.16b
	str	q2, [sp, 240]
	tbl	v19.16b, {v12.16b - v15.16b}, v18.16b
	ld1	{v24.16b - v27.16b}, [x0]
	add	x0, sp, 128
	mov	v24.16b, v4.16b
	mov	v25.16b, v16.16b
	mov	v26.16b, v5.16b
	st1	{v8.16b - v11.16b}, [x0]
	add	x0, sp, 256
	mov	v30.16b, v2.16b
	str	q16, [sp, 176]
	mov	v31.16b, v7.16b
	st1	{v20.16b - v23.16b}, [x0]
	adrp	x0, .LC0
	add	x0, x0, :lo12:.LC0
	str	q4, [sp, 304]
	ld1	{v20.16b - v23.16b}, [x0]
	add	x0, sp, 320
	mov	v20.16b, v2.16b
	mov	v21.16b, v7.16b
	st1	{v24.16b - v27.16b}, [x0]
	add	x0, sp, 64
	ld1	{v8.16b - v11.16b}, [x0]
	add	x0, sp, 128
	tbl	v18.16b, {v8.16b - v11.16b}, v18.16b
	tbl	v8.16b, {v28.16b - v31.16b}, v0.16b
	ld1	{v9.16b - v12.16b}, [x0]
	adrp	x0, .LC0
	add	x0, x0, :lo12:.LC0
	stp	q19, q18, [x2]
	tbl	v12.16b, {v9.16b - v12.16b}, v0.16b
	ld1	{v28.16b - v31.16b}, [x0]
	add	x0, sp, 192
	add	v0.16b, v1.16b, v0.16b
	str	q3, [sp, 368]
	ld1	{v24.16b - v27.16b}, [x0]
	add	x0, sp, 64
	mov	v28.16b, v6.16b
	mov	v29.16b, v4.16b
	st1	{v20.16b - v23.16b}, [x0]
	add	x0, sp, 256
	tbl	v9.16b, {v24.16b - v27.16b}, v0.16b
	mov	v30.16b, v16.16b
	stp	q8, q12, [x2, 32]
	ld1	{v10.16b - v13.16b}, [x0]
	adrp	x0, .LC0
	mov	v24.16b, v3.16b
	add	x0, x0, :lo12:.LC0
	mov	v25.16b, v2.16b
	tbl	v8.16b, {v10.16b - v13.16b}, v0.16b
	ld1	{v20.16b - v23.16b}, [x0]
	adrp	x0, .LC0
	add	v0.16b, v1.16b, v0.16b
	add	x0, x0, :lo12:.LC0
	mov	v20.16b, v17.16b
	mov	v21.16b, v6.16b
	ld1	{v12.16b - v15.16b}, [x0]
	add	x0, sp, 320
	mov	v22.16b, v4.16b
	stp	q9, q8, [x2, 64]
	ld1	{v8.16b - v11.16b}, [x0]
	adrp	x0, .LC0
	add	x0, x0, :lo12:.LC0
	str	q17, [sp, 96]
	tbl	v18.16b, {v8.16b - v11.16b}, v0.16b
	ld1	{v8.16b - v11.16b}, [x0]
	add	x0, sp, 192
	mov	v12.16b, v5.16b
	mov	v13.16b, v3.16b
	mov	v14.16b, v2.16b
	st1	{v8.16b - v11.16b}, [x0]
	add	x0, sp, 128
	mov	v26.16b, v7.16b
	str	q2, [sp, 192]
	mov	v27.16b, v17.16b
	str	q18, [x2, 96]
	st1	{v28.16b - v31.16b}, [x0]
	add	x0, sp, 192
	str	q6, [sp, 112]
	ld1	{v8.16b - v11.16b}, [x0]
	add	x0, sp, 192
	mov	v10.16b, v17.16b
	str	q5, [sp, 176]
	mov	v29.16b, v16.16b
	st1	{v20.16b - v23.16b}, [x0]
	add	x0, sp, 256
	mov	v11.16b, v6.16b
	mov	v30.16b, v5.16b
	mov	v9.16b, v7.16b
	st1	{v12.16b - v15.16b}, [x0]
	add	x0, sp, 64
	mov	v13.16b, v17.16b
	mov	v14.16b, v6.16b
	ld1	{v18.16b - v21.16b}, [x0]
	add	x0, sp, 128
	mov	v15.16b, v4.16b
	mov	v28.16b, v4.16b
	tbl	v18.16b, {v18.16b - v21.16b}, v0.16b
	add	v0.16b, v1.16b, v0.16b
	mov	v12.16b, v7.16b
	mov	v31.16b, v3.16b
	mov	v22.16b, v3.16b
	str	q18, [x2, 112]
	tbl	v17.16b, {v24.16b - v27.16b}, v0.16b
	ld1	{v18.16b - v21.16b}, [x0]
	add	x0, sp, 192
	mov	v23.16b, v2.16b
	str	q16, [sp, 240]
	tbl	v18.16b, {v18.16b - v21.16b}, v0.16b
	add	v0.16b, v1.16b, v0.16b
	mov	v20.16b, v16.16b
	mov	v21.16b, v5.16b
	str	q18, [x2, 128]
	str	q7, [sp, 304]
	str	q17, [x2, 144]
	ld1	{v16.16b - v19.16b}, [x0]
	add	x0, sp, 256
	tbl	v6.16b, {v16.16b - v19.16b}, v0.16b
	ld1	{v16.16b - v19.16b}, [x0]
	tbl	v5.16b, {v16.16b - v19.16b}, v0.16b
	add	v0.16b, v1.16b, v0.16b
	stp	q6, q5, [x2, 160]
	tbl	v12.16b, {v12.16b - v15.16b}, v0.16b
	add	v1.16b, v1.16b, v0.16b
	tbl	v20.16b, {v20.16b - v23.16b}, v0.16b
	str	q12, [x2, 192]
	tbl	v8.16b, {v8.16b - v11.16b}, v1.16b
	tbl	v28.16b, {v28.16b - v31.16b}, v1.16b
	str	q20, [x2, 208]
	stp	q8, q28, [x2, 224]
	ldp	d10, d11, [sp, 16]
	ldp	d12, d13, [sp, 32]
	ldp	d14, d15, [sp, 48]
	ldp	d8, d9, [sp], 384
	.cfi_restore 73
	.cfi_restore 72
	.cfi_restore 78
	.cfi_restore 79
	.cfi_restore 76
	.cfi_restore 77
	.cfi_restore 74
	.cfi_restore 75
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE6665:
	.size	_Z7stepTLBPsPimmPa.constprop.0, .-_Z7stepTLBPsPimmPa.constprop.0
	.align	2
	.p2align 4,,11
	.type	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0, %function
_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0:
.LFB6666:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	mov	x29, sp
	stp	x19, x20, [sp, 16]
	.cfi_offset 19, -16
	.cfi_offset 20, -8
	mov	x19, x0
	ldr	x0, [x0]
	ldr	x0, [x0, -24]
	add	x0, x19, x0
	ldr	x20, [x0, 240]
	cbz	x20, .L11
	ldrb	w0, [x20, 56]
	cbz	w0, .L7
	ldrb	w1, [x20, 67]
.L8:
	mov	x0, x19
	bl	_ZNSo3putEc
	ldp	x19, x20, [sp, 16]
	ldp	x29, x30, [sp], 32
	.cfi_remember_state
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	b	_ZNSo5flushEv
.L7:
	.cfi_restore_state
	mov	x0, x20
	bl	_ZNKSt5ctypeIcE13_M_widen_initEv
	ldr	x1, [x20]
	adrp	x0, _ZNKSt5ctypeIcE8do_widenEc
	add	x0, x0, :lo12:_ZNKSt5ctypeIcE8do_widenEc
	ldr	x2, [x1, 48]
	cmp	x2, x0
	bne	.L12
	mov	w1, 10
	b	.L8
.L12:
	mov	w1, 10
	mov	x0, x20
	blr	x2
	and	w1, w0, 255
	b	.L8
.L11:
	bl	_ZSt16__throw_bad_castv
	.cfi_endproc
.LFE6666:
	.size	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0, .-_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
	.section	.rodata.str1.8,"aMS",@progbits,1
	.align	3
.LC1:
	.string	" = ["
	.align	3
.LC2:
	.string	"\t[ "
	.align	3
.LC3:
	.string	", "
	.align	3
.LC4:
	.string	"],"
	.align	3
.LC5:
	.string	"]"
	.text
	.align	2
	.p2align 4,,11
	.type	_Z5printIiEvPT_mmNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb.isra.0, %function
_Z5printIiEvPT_mmNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb.isra.0:
.LFB6669:
	.cfi_startproc
	stp	x29, x30, [sp, -112]!
	.cfi_def_cfa_offset 112
	.cfi_offset 29, -112
	.cfi_offset 30, -104
	mov	x29, sp
	stp	x23, x24, [sp, 48]
	.cfi_offset 23, -64
	.cfi_offset 24, -56
	adrp	x23, :got:_ZSt4cout
	and	w24, w5, 255
	stp	x21, x22, [sp, 32]
	.cfi_offset 21, -80
	.cfi_offset 22, -72
	mov	x21, x2
	mov	x2, x4
	ldr	x22, [x23, #:got_lo12:_ZSt4cout]
	stp	x19, x20, [sp, 16]
	.cfi_offset 19, -96
	.cfi_offset 20, -88
	mov	x19, x0
	mov	x0, x22
	stp	x25, x26, [sp, 64]
	.cfi_offset 25, -48
	.cfi_offset 26, -40
	mov	x25, x1
	mov	x1, x3
	str	w24, [sp, 108]
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	mov	x20, x0
	adrp	x1, .LC1
	add	x1, x1, :lo12:.LC1
	mov	x2, 4
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	mov	x0, x20
	bl	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
	cbnz	w24, .L33
	cbz	x25, .L16
.L15:
	adrp	x22, .LC3
	add	x22, x22, :lo12:.LC3
	stp	x27, x28, [sp, 80]
	.cfi_offset 28, -24
	.cfi_offset 27, -32
	lsl	x28, x21, 2
	adrp	x0, _ZNKSt5ctypeIcE8do_widenEc
	mov	x24, 0
	add	x0, x0, :lo12:_ZNKSt5ctypeIcE8do_widenEc
	adrp	x27, .LC2
	str	x0, [sp, 96]
	.p2align 3,,7
.L22:
	ldr	x20, [x23, #:got_lo12:_ZSt4cout]
	add	x1, x27, :lo12:.LC2
	mov	x2, 3
	mov	x0, x20
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	cbz	x21, .L17
	mov	x26, 0
	.p2align 3,,7
.L18:
	ldr	w1, [x19, x26, lsl 2]
	mov	x0, x20
	add	x26, x26, 1
	bl	_ZNSo9_M_insertImEERSoT_
	mov	x1, x22
	mov	x2, 2
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	cmp	x21, x26
	bne	.L18
.L17:
	ldr	x20, [x23, #:got_lo12:_ZSt4cout]
	adrp	x0, .LC4
	mov	x2, 2
	add	x1, x0, :lo12:.LC4
	mov	x0, x20
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	ldr	x0, [x20]
	ldr	x0, [x0, -24]
	add	x0, x0, x20
	ldr	x20, [x0, 240]
	cbz	x20, .L34
	ldrb	w0, [x20, 56]
	cbz	w0, .L20
	ldrb	w1, [x20, 67]
.L21:
	ldr	x0, [x23, #:got_lo12:_ZSt4cout]
	add	x24, x24, 1
	add	x19, x19, x28
	bl	_ZNSo3putEc
	bl	_ZNSo5flushEv
	cmp	x24, x25
	bcc	.L22
	ldr	w0, [sp, 108]
	ldp	x27, x28, [sp, 80]
	.cfi_restore 28
	.cfi_restore 27
	cbnz	w0, .L23
.L16:
	ldr	x23, [x23, #:got_lo12:_ZSt4cout]
	mov	x2, 1
	adrp	x1, .LC5
	add	x1, x1, :lo12:.LC5
	mov	x0, x23
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	mov	x0, x23
	ldp	x19, x20, [sp, 16]
	ldp	x21, x22, [sp, 32]
	ldp	x23, x24, [sp, 48]
	ldp	x25, x26, [sp, 64]
	ldp	x29, x30, [sp], 112
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	b	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
	.p2align 2,,3
.L20:
	.cfi_def_cfa_offset 112
	.cfi_offset 19, -96
	.cfi_offset 20, -88
	.cfi_offset 21, -80
	.cfi_offset 22, -72
	.cfi_offset 23, -64
	.cfi_offset 24, -56
	.cfi_offset 25, -48
	.cfi_offset 26, -40
	.cfi_offset 27, -32
	.cfi_offset 28, -24
	.cfi_offset 29, -112
	.cfi_offset 30, -104
	mov	x0, x20
	bl	_ZNKSt5ctypeIcE13_M_widen_initEv
	ldr	x0, [x20]
	mov	w1, 10
	ldr	x2, [x0, 48]
	ldr	x0, [sp, 96]
	cmp	x2, x0
	beq	.L21
	mov	x0, x20
	blr	x2
	and	w1, w0, 255
	b	.L21
.L33:
	.cfi_restore 27
	.cfi_restore 28
	ldr	x0, [x22]
	mov	w2, -75
	ldr	x1, [x0, -24]
	add	x1, x1, x22
	ldr	w0, [x1, 24]
	and	w0, w0, w2
	orr	w0, w0, 8
	str	w0, [x1, 24]
	cbnz	x25, .L15
.L23:
	ldr	x0, [x23, #:got_lo12:_ZSt4cout]
	mov	w2, -75
	ldr	x1, [x0]
	ldr	x1, [x1, -24]
	add	x1, x1, x0
	ldr	w0, [x1, 24]
	and	w0, w0, w2
	orr	w0, w0, 2
	str	w0, [x1, 24]
	b	.L16
.L34:
	.cfi_offset 27, -32
	.cfi_offset 28, -24
	bl	_ZSt16__throw_bad_castv
	.cfi_endproc
.LFE6669:
	.size	_Z5printIiEvPT_mmNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb.isra.0, .-_Z5printIiEvPT_mmNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb.isra.0
	.align	2
	.p2align 4,,11
	.global	_Z7stepTLBPsPimmPa
	.type	_Z7stepTLBPsPimmPa, %function
_Z7stepTLBPsPimmPa:
.LFB6150:
	.cfi_startproc
	lsl	x3, x3, 1
	mov	x2, x1
	add	x1, x0, x3
	stp	d8, d9, [sp, -384]!
	.cfi_def_cfa_offset 384
	.cfi_offset 72, -384
	.cfi_offset 73, -376
	add	x4, x1, x3
	add	x5, x4, x3
	stp	d10, d11, [sp, 16]
	add	x7, x5, x3
	add	x6, x7, x3
	stp	d12, d13, [sp, 32]
	add	x8, x6, x3
	stp	d14, d15, [sp, 48]
	.cfi_offset 74, -368
	.cfi_offset 75, -360
	.cfi_offset 76, -352
	.cfi_offset 77, -344
	.cfi_offset 78, -336
	.cfi_offset 79, -328
	ldr	q4, [x7, x3]
	ldr	q5, [x8, x3]
	ldr	q16, [x6, x3]
	ldr	q6, [x5, x3]
	ldr	q17, [x4, x3]
	ldr	q7, [x1, x3]
	ldr	q2, [x0, x3]
	ldr	q3, [x0]
#APP
// 23 "barrel-shift-w8a8-using-tlb.cc" 1
	mov w1, 0xFFFF
	mov w0, 0x0100
	ins v18.h[0], w0
	ins v18.h[1], w1
	mov w0, 0x1110
	ins v18.h[2], w0
	ins v18.h[3], w1
	mov w0, 0x2120
	ins v18.h[4], w0
	ins v18.h[5], w1
	mov w0, 0x3130
	ins v18.h[6], w0
	ins v18.h[7], w1
	mov w0, 0x02
	dup v1.16b, w0
	mov w1, 0x0000
	ins v1.h[1], w1
	ins v1.h[3], w1
	ins v1.h[5], w1
	ins v1.h[7], w1
	
// 0 "" 2
#NO_APP
	mov	v20.16b, v17.16b
	adrp	x0, .LC0
	mov	v8.16b, v6.16b
	add	x0, x0, :lo12:.LC0
	mov	v9.16b, v4.16b
	add	v0.16b, v18.16b, v1.16b
	ld1	{v12.16b - v15.16b}, [x0]
	adrp	x0, .LC0
	mov	v12.16b, v3.16b
	add	x0, x0, :lo12:.LC0
	mov	v28.16b, v5.16b
	mov	v21.16b, v13.16b
	mov	v22.16b, v14.16b
	mov	v23.16b, v15.16b
	ld1	{v24.16b - v27.16b}, [x0]
	add	x0, sp, 128
	mov	v10.16b, v14.16b
	mov	v11.16b, v15.16b
	mov	v24.16b, v16.16b
	st1	{v20.16b - v23.16b}, [x0]
	adrp	x0, .LC0
	add	x0, x0, :lo12:.LC0
	str	q6, [sp, 144]
	mov	v25.16b, v5.16b
	ld1	{v20.16b - v23.16b}, [x0]
	add	x0, sp, 64
	mov	v20.16b, v7.16b
	mov	v21.16b, v17.16b
	mov	v22.16b, v6.16b
	st1	{v8.16b - v11.16b}, [x0]
	add	x0, sp, 128
	mov	v13.16b, v2.16b
	stp	q16, q5, [sp, 96]
	ld1	{v8.16b - v11.16b}, [x0]
	st1	{v24.16b - v27.16b}, [x0]
	adrp	x0, .LC0
	add	x0, x0, :lo12:.LC0
	str	q3, [sp, 160]
	mov	v10.16b, v4.16b
	str	q2, [sp, 176]
	ld1	{v24.16b - v27.16b}, [x0]
	add	x0, sp, 320
	mov	v24.16b, v4.16b
	mov	v25.16b, v16.16b
	mov	v26.16b, v5.16b
	st1	{v8.16b - v11.16b}, [x0]
	add	x0, sp, 256
	mov	v14.16b, v7.16b
	str	q16, [sp, 368]
	mov	v15.16b, v17.16b
	st1	{v20.16b - v23.16b}, [x0]
	adrp	x0, .LC0
	add	x0, x0, :lo12:.LC0
	str	q4, [sp, 304]
	mov	v29.16b, v3.16b
	ld1	{v20.16b - v23.16b}, [x0]
	add	x0, sp, 192
	mov	v30.16b, v2.16b
	mov	v31.16b, v7.16b
	st1	{v24.16b - v27.16b}, [x0]
	add	x0, sp, 64
	tbl	v19.16b, {v12.16b - v15.16b}, v18.16b
	mov	v20.16b, v2.16b
	ld1	{v8.16b - v11.16b}, [x0]
	add	x0, sp, 320
	mov	v21.16b, v7.16b
	tbl	v18.16b, {v8.16b - v11.16b}, v18.16b
	tbl	v8.16b, {v28.16b - v31.16b}, v0.16b
	ld1	{v9.16b - v12.16b}, [x0]
	adrp	x0, .LC0
	add	x0, x0, :lo12:.LC0
	stp	q19, q18, [x2]
	tbl	v12.16b, {v9.16b - v12.16b}, v0.16b
	ld1	{v28.16b - v31.16b}, [x0]
	add	x0, sp, 128
	add	v0.16b, v1.16b, v0.16b
	str	q3, [sp, 240]
	ld1	{v24.16b - v27.16b}, [x0]
	add	x0, sp, 128
	mov	v28.16b, v6.16b
	mov	v29.16b, v4.16b
	st1	{v20.16b - v23.16b}, [x0]
	add	x0, sp, 256
	tbl	v9.16b, {v24.16b - v27.16b}, v0.16b
	mov	v30.16b, v16.16b
	stp	q8, q12, [x2, 32]
	ld1	{v10.16b - v13.16b}, [x0]
	adrp	x0, .LC0
	mov	v24.16b, v3.16b
	add	x0, x0, :lo12:.LC0
	mov	v25.16b, v2.16b
	tbl	v8.16b, {v10.16b - v13.16b}, v0.16b
	ld1	{v20.16b - v23.16b}, [x0]
	adrp	x0, .LC0
	add	v0.16b, v1.16b, v0.16b
	add	x0, x0, :lo12:.LC0
	mov	v20.16b, v17.16b
	mov	v21.16b, v6.16b
	ld1	{v12.16b - v15.16b}, [x0]
	add	x0, sp, 192
	mov	v22.16b, v4.16b
	stp	q9, q8, [x2, 64]
	ld1	{v8.16b - v11.16b}, [x0]
	adrp	x0, .LC0
	add	x0, x0, :lo12:.LC0
	str	q17, [sp, 160]
	tbl	v18.16b, {v8.16b - v11.16b}, v0.16b
	ld1	{v8.16b - v11.16b}, [x0]
	add	x0, sp, 64
	mov	v12.16b, v5.16b
	mov	v13.16b, v3.16b
	mov	v14.16b, v2.16b
	st1	{v8.16b - v11.16b}, [x0]
	add	x0, sp, 128
	mov	v26.16b, v7.16b
	str	q2, [sp, 64]
	ld1	{v8.16b - v11.16b}, [x0]
	add	x0, sp, 256
	mov	v27.16b, v17.16b
	st1	{v8.16b - v11.16b}, [x0]
	add	x0, sp, 192
	str	q18, [x2, 96]
	st1	{v28.16b - v31.16b}, [x0]
	add	x0, sp, 64
	str	q6, [sp, 304]
	ld1	{v8.16b - v11.16b}, [x0]
	add	x0, sp, 64
	mov	v10.16b, v17.16b
	str	q5, [sp, 240]
	mov	v29.16b, v16.16b
	st1	{v20.16b - v23.16b}, [x0]
	add	x0, sp, 128
	mov	v11.16b, v6.16b
	mov	v30.16b, v5.16b
	mov	v9.16b, v7.16b
	st1	{v12.16b - v15.16b}, [x0]
	add	x0, sp, 256
	mov	v13.16b, v17.16b
	mov	v14.16b, v6.16b
	ld1	{v18.16b - v21.16b}, [x0]
	add	x0, sp, 192
	mov	v15.16b, v4.16b
	mov	v28.16b, v4.16b
	tbl	v18.16b, {v18.16b - v21.16b}, v0.16b
	add	v0.16b, v1.16b, v0.16b
	mov	v12.16b, v7.16b
	mov	v31.16b, v3.16b
	mov	v22.16b, v3.16b
	str	q18, [x2, 112]
	tbl	v17.16b, {v24.16b - v27.16b}, v0.16b
	ld1	{v18.16b - v21.16b}, [x0]
	add	x0, sp, 64
	mov	v23.16b, v2.16b
	str	q16, [sp, 112]
	tbl	v18.16b, {v18.16b - v21.16b}, v0.16b
	add	v0.16b, v1.16b, v0.16b
	mov	v20.16b, v16.16b
	mov	v21.16b, v5.16b
	str	q18, [x2, 128]
	str	q7, [sp, 176]
	str	q17, [x2, 144]
	ld1	{v16.16b - v19.16b}, [x0]
	add	x0, sp, 128
	tbl	v6.16b, {v16.16b - v19.16b}, v0.16b
	ld1	{v16.16b - v19.16b}, [x0]
	tbl	v5.16b, {v16.16b - v19.16b}, v0.16b
	add	v0.16b, v1.16b, v0.16b
	stp	q6, q5, [x2, 160]
	tbl	v12.16b, {v12.16b - v15.16b}, v0.16b
	add	v1.16b, v1.16b, v0.16b
	tbl	v20.16b, {v20.16b - v23.16b}, v0.16b
	tbl	v8.16b, {v8.16b - v11.16b}, v1.16b
	tbl	v28.16b, {v28.16b - v31.16b}, v1.16b
	stp	q12, q20, [x2, 192]
	stp	q8, q28, [x2, 224]
	ldp	d10, d11, [sp, 16]
	ldp	d12, d13, [sp, 32]
	ldp	d14, d15, [sp, 48]
	ldp	d8, d9, [sp], 384
	.cfi_restore 73
	.cfi_restore 72
	.cfi_restore 78
	.cfi_restore 79
	.cfi_restore 76
	.cfi_restore 77
	.cfi_restore 74
	.cfi_restore 75
	.cfi_def_cfa_offset 0
	ret
	.cfi_endproc
.LFE6150:
	.size	_Z7stepTLBPsPimmPa, .-_Z7stepTLBPsPimmPa
	.align	2
	.p2align 4,,11
	.global	_Z12stepOriginalPsPimmPa
	.type	_Z12stepOriginalPsPimmPa, %function
_Z12stepOriginalPsPimmPa:
.LFB6151:
	.cfi_startproc
	lsl	x11, x3, 1
	lsl	x14, x3, 2
	add	x13, x11, x3
	lsl	x12, x3, 3
	add	x16, x14, x3
	sub	x12, x12, x3
	lsl	x15, x13, 1
	mov	x4, 0
	.p2align 3,,7
.L38:
	ldrsh	w5, [x0]
	mov	x2, x4
	str	w5, [x1, x4, lsl 2]
	add	x4, x4, 1
	add	x9, x2, 2
	and	x10, x4, 7
	add	x10, x10, x3
	add	x8, x2, 3
	and	x9, x9, 7
	add	x7, x2, 4
	add	x9, x9, x11
	and	x8, x8, 7
	ldrsh	w18, [x0, 2]
	add	x8, x8, x13
	add	x6, x2, 5
	and	x7, x7, 7
	ldrsh	w17, [x0, 4]
	add	x7, x7, x14
	add	x5, x2, 6
	and	x6, x6, 7
	str	w18, [x1, x10, lsl 2]
	add	x6, x6, x16
	ldrsh	w10, [x0, 6]
	and	x5, x5, 7
	add	x2, x2, 7
	str	w17, [x1, x9, lsl 2]
	add	x5, x5, x15
	ldrsh	w9, [x0, 8]
	and	x2, x2, 7
	str	w10, [x1, x8, lsl 2]
	add	x2, x2, x12
	ldrsh	w8, [x0, 10]
	str	w9, [x1, x7, lsl 2]
	ldrsh	w7, [x0, 12]
	str	w8, [x1, x6, lsl 2]
	ldrsh	w6, [x0, 14]
	add	x0, x0, x11
	str	w7, [x1, x5, lsl 2]
	str	w6, [x1, x2, lsl 2]
	cmp	x4, 8
	bne	.L38
	ret
	.cfi_endproc
.LFE6151:
	.size	_Z12stepOriginalPsPimmPa, .-_Z12stepOriginalPsPimmPa
	.section	.rodata._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_.str1.8,"aMS",@progbits,1
	.align	3
.LC6:
	.string	"basic_string::_M_construct null not valid"
	.section	.text._ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_,"axG",@progbits,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC5IS3_EEPKcRKS3_,comdat
	.align	2
	.p2align 4,,11
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
	.type	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_, %function
_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_:
.LFB6411:
	.cfi_startproc
	stp	x29, x30, [sp, -64]!
	.cfi_def_cfa_offset 64
	.cfi_offset 29, -64
	.cfi_offset 30, -56
	adrp	x2, :got:__stack_chk_guard
	mov	x29, sp
	ldr	x2, [x2, #:got_lo12:__stack_chk_guard]
	stp	x19, x20, [sp, 16]
	.cfi_offset 19, -48
	.cfi_offset 20, -40
	mov	x20, x0
	stp	x21, x22, [sp, 32]
	.cfi_offset 21, -32
	.cfi_offset 22, -24
	add	x22, x0, 16
	ldr	x0, [x2]
	str	x0, [sp, 56]
	mov	x0, 0
	str	x22, [x20]
	cbz	x1, .L41
	mov	x21, x1
	mov	x0, x1
	bl	strlen
	str	x0, [sp, 48]
	mov	x19, x0
	cmp	x0, 15
	bhi	.L53
	cmp	x0, 1
	bne	.L45
	ldrb	w0, [x21]
	strb	w0, [x20, 16]
.L46:
	adrp	x0, :got:__stack_chk_guard
	str	x19, [x20, 8]
	strb	wzr, [x22, x19]
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x2, [sp, 56]
	ldr	x1, [x0]
	subs	x2, x2, x1
	mov	x1, 0
	bne	.L54
	ldp	x19, x20, [sp, 16]
	ldp	x21, x22, [sp, 32]
	ldp	x29, x30, [sp], 64
	.cfi_remember_state
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa_offset 0
	ret
	.p2align 2,,3
.L45:
	.cfi_restore_state
	cbz	x0, .L46
	b	.L44
	.p2align 2,,3
.L53:
	add	x1, sp, 48
	mov	x0, x20
	mov	x2, 0
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEE9_M_createERmm
	mov	x22, x0
	str	x0, [x20]
	ldr	x1, [sp, 48]
	str	x1, [x20, 16]
.L44:
	mov	x2, x19
	mov	x0, x22
	mov	x1, x21
	bl	memcpy
	ldr	x22, [x20]
	ldr	x19, [sp, 48]
	b	.L46
.L54:
	bl	__stack_chk_fail
.L41:
	adrp	x0, .LC6
	add	x0, x0, :lo12:.LC6
	bl	_ZSt19__throw_logic_errorPKc
	.cfi_endproc
.LFE6411:
	.size	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_, .-_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
	.weak	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1IS3_EEPKcRKS3_
	.set	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1IS3_EEPKcRKS3_,_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC2IS3_EEPKcRKS3_
	.section	.rodata.str1.8
	.align	3
.LC7:
	.string	"Sanity Check FAILED"
	.align	3
.LC8:
	.string	"i"
	.align	3
.LC9:
	.string	"Output Original"
	.align	3
.LC10:
	.string	"Output New"
	.align	3
.LC11:
	.string	"Sanity Check PASS"
	.global	__extenddftf2
	.global	__floatditf
	.global	__addtf3
	.global	__subtf3
	.global	__trunctfdf2
	.align	3
.LC12:
	.string	"Original Time for "
	.align	3
.LC13:
	.string	" iterations: "
	.align	3
.LC14:
	.string	" seconds."
	.align	3
.LC15:
	.string	"   TLB   Time for "
	.section	.text.startup,"ax",@progbits
	.align	2
	.p2align 4,,11
	.global	main
	.type	main, %function
main:
.LFB6157:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA6157
	stp	x29, x30, [sp, -224]!
	.cfi_def_cfa_offset 224
	.cfi_offset 29, -224
	.cfi_offset 30, -216
	adrp	x1, :got:__stack_chk_guard
	mov	x0, 128
	mov	x29, sp
	ldr	x1, [x1, #:got_lo12:__stack_chk_guard]
	stp	x19, x20, [sp, 16]
	stp	x21, x22, [sp, 32]
	stp	x23, x24, [sp, 48]
	stp	x25, x26, [sp, 64]
	stp	d8, d9, [sp, 96]
	ldr	x2, [x1]
	str	x2, [sp, 216]
	mov	x2, 0
.LEHB0:
	.cfi_offset 19, -208
	.cfi_offset 20, -200
	.cfi_offset 21, -192
	.cfi_offset 22, -184
	.cfi_offset 23, -176
	.cfi_offset 24, -168
	.cfi_offset 25, -160
	.cfi_offset 26, -152
	.cfi_offset 72, -128
	.cfi_offset 73, -120
	bl	_Znam
	mov	x20, x0
	mov	x0, 256
	bl	_Znam
	mov	x19, x0
	mov	x0, 256
	bl	_Znam
	mov	x21, x0
	adrp	x2, .LC16
	adrp	x0, .LC17
	mov	x1, x19
	ldr	q2, [x2, #:lo12:.LC16]
	adrp	x2, .LC18
	ldr	q1, [x0, #:lo12:.LC17]
	mov	x0, x20
	ldr	q0, [x2, #:lo12:.LC18]
	adrp	x2, .LC19
	stp	q2, q1, [x20]
	ldr	q2, [x2, #:lo12:.LC19]
	adrp	x2, .LC20
	str	q0, [x20, 32]
	ldr	q1, [x2, #:lo12:.LC20]
	adrp	x2, .LC21
	ldr	q0, [x2, #:lo12:.LC21]
	adrp	x2, .LC22
	stp	q2, q1, [x20, 48]
	ldr	q2, [x2, #:lo12:.LC22]
	adrp	x2, .LC23
	str	q0, [x20, 80]
	ldr	q1, [x2, #:lo12:.LC23]
	movi	v0.4s, 0
	stp	q2, q1, [x20, 96]
	stp	q0, q0, [x19]
	stp	q0, q0, [x21]
	stp	q0, q0, [x19, 32]
	stp	q0, q0, [x21, 32]
	stp	q0, q0, [x19, 64]
	stp	q0, q0, [x21, 64]
	stp	q0, q0, [x19, 96]
	stp	q0, q0, [x21, 96]
	stp	q0, q0, [x19, 128]
	stp	q0, q0, [x21, 128]
	stp	q0, q0, [x19, 160]
	stp	q0, q0, [x21, 160]
	stp	q0, q0, [x19, 192]
	stp	q0, q0, [x21, 192]
	stp	q0, q0, [x19, 224]
	stp	q0, q0, [x21, 224]
	bl	_Z7stepTLBPsPimmPa.constprop.0
	mov	x1, x20
	mov	x2, 0
.L56:
	ldrsh	w3, [x1]
	mov	x0, x2
	str	w3, [x21, x2, lsl 2]
	add	x2, x2, 1
	add	x7, x0, 2
	and	x8, x2, 7
	add	x8, x8, 8
	add	x6, x0, 3
	and	x7, x7, 7
	add	x5, x0, 4
	add	x7, x7, 16
	and	x6, x6, 7
	ldrsh	w10, [x1, 2]
	add	x6, x6, 24
	add	x4, x0, 5
	and	x5, x5, 7
	ldrsh	w9, [x1, 4]
	add	x5, x5, 32
	add	x3, x0, 6
	and	x4, x4, 7
	str	w10, [x21, x8, lsl 2]
	add	x4, x4, 40
	ldrsh	w8, [x1, 6]
	and	x3, x3, 7
	add	x0, x0, 7
	str	w9, [x21, x7, lsl 2]
	add	x3, x3, 48
	ldrsh	w7, [x1, 8]
	and	x0, x0, 7
	str	w8, [x21, x6, lsl 2]
	add	x0, x0, 56
	ldrsh	w6, [x1, 10]
	str	w7, [x21, x5, lsl 2]
	add	x1, x1, 16
	ldrsh	w5, [x1, -4]
	str	w6, [x21, x4, lsl 2]
	ldrsh	w4, [x1, -2]
	str	w5, [x21, x3, lsl 2]
	str	w4, [x21, x0, lsl 2]
	cmp	x2, 8
	bne	.L56
	mov	x0, x19
	mov	x1, x21
	add	x5, x19, 256
	mov	w2, 1
.L57:
	ldp	w6, w3, [x1, 8]
	ldp	w7, w4, [x0, 8]
	add	x0, x0, 32
	ldr	w8, [x0, -32]
	cmp	w7, w6
	ldr	w6, [x0, -28]
	ccmp	w4, w3, 0, eq
	ldr	w9, [x0, -16]
	ldp	w7, w4, [x1]
	cset	w3, eq
	cmp	w8, w7
	ccmp	w6, w4, 0, eq
	ldr	w6, [x0, -12]
	ldp	w8, w4, [x1, 16]
	cset	w7, eq
	and	w3, w3, w7
	ldr	w7, [x0, -4]
	cmp	w9, w8
	ldr	w9, [x0, -8]
	ccmp	w6, w4, 0, eq
	ldp	w8, w6, [x1, 24]
	cset	w4, eq
	and	w4, w4, w3
	add	x1, x1, 32
	cmp	w9, w8
	ccmp	w7, w6, 0, eq
	cset	w3, eq
	and	w3, w3, w4
	and	w2, w2, w3
	cmp	x0, x5
	bne	.L57
	cbnz	w2, .L58
	adrp	x22, :got:_ZSt4cout
	adrp	x1, .LC7
	add	x1, x1, :lo12:.LC7
	mov	x2, 19
	ldr	x23, [x22, #:got_lo12:_ZSt4cout]
	stp	x27, x28, [sp, 80]
	.cfi_offset 28, -136
	.cfi_offset 27, -144
	add	x25, sp, 160
	mov	x0, x23
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	mov	x0, x23
	add	x28, sp, 184
	bl	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
	mov	x2, x25
	mov	x0, x28
	adrp	x1, .LC8
	add	x1, x1, :lo12:.LC8
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1IS3_EEPKcRKS3_
.LEHE0:
	ldp	x1, x2, [sp, 184]
	mov	x0, x23
.LEHB1:
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	adrp	x1, .LC1
	mov	x23, x0
	add	x1, x1, :lo12:.LC1
	mov	x2, 4
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	mov	x0, x23
	bl	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
	adrp	x26, .LC3
	adrp	x0, .LC4
	add	x26, x26, :lo12:.LC3
	add	x0, x0, :lo12:.LC4
	mov	x24, x20
	str	x0, [sp, 128]
	add	x0, x20, 128
	str	x0, [sp, 112]
.L63:
	ldr	x0, [x22, #:got_lo12:_ZSt4cout]
	adrp	x1, .LC2
	mov	x2, 3
	add	x1, x1, :lo12:.LC2
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	ldr	x27, [x22, #:got_lo12:_ZSt4cout]
	mov	x23, 0
.L59:
	ldrsh	w1, [x24, x23, lsl 1]
	mov	x0, x27
	uxtw	x1, w1
	bl	_ZNSo9_M_insertImEERSoT_
	mov	x1, x26
	mov	x2, 2
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	add	x23, x23, 1
	cmp	x23, 8
	bne	.L59
	ldr	x0, [x22, #:got_lo12:_ZSt4cout]
	mov	x2, 2
	ldr	x1, [sp, 128]
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	ldr	x1, [x22, #:got_lo12:_ZSt4cout]
	ldr	x0, [x1]
	ldr	x0, [x0, -24]
	add	x0, x0, x1
	ldr	x23, [x0, 240]
	cbz	x23, .L95
	ldrb	w0, [x23, 56]
	cbz	w0, .L61
	ldrb	w1, [x23, 67]
.L62:
	ldr	x0, [x22, #:got_lo12:_ZSt4cout]
	bl	_ZNSo3putEc
	bl	_ZNSo5flushEv
	ldr	x0, [sp, 112]
	add	x24, x24, 16
	cmp	x24, x0
	bne	.L63
	ldr	x0, [x22, #:got_lo12:_ZSt4cout]
	adrp	x1, .LC5
	mov	x2, 1
	add	x1, x1, :lo12:.LC5
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	ldr	x0, [x22, #:got_lo12:_ZSt4cout]
	bl	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE1:
	ldr	x0, [sp, 184]
	add	x23, sp, 200
	cmp	x0, x23
	beq	.L64
	ldr	x1, [sp, 200]
	add	x1, x1, 1
	bl	_ZdlPvm
.L64:
	mov	x2, x25
	mov	x0, x28
	adrp	x1, .LC9
	add	x1, x1, :lo12:.LC9
.LEHB2:
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1IS3_EEPKcRKS3_
.LEHE2:
	ldp	x3, x4, [sp, 184]
	mov	x2, 8
	mov	x0, x21
	mov	x1, x2
	mov	w5, 0
.LEHB3:
	bl	_Z5printIiEvPT_mmNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb.isra.0
.LEHE3:
	ldr	x0, [sp, 184]
	cmp	x0, x23
	beq	.L65
	ldr	x1, [sp, 200]
	add	x1, x1, 1
	bl	_ZdlPvm
.L65:
	mov	x0, x28
	mov	x2, x25
	adrp	x1, .LC10
	add	x1, x1, :lo12:.LC10
.LEHB4:
	bl	_ZNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEC1IS3_EEPKcRKS3_
.LEHE4:
	ldp	x3, x4, [sp, 184]
	mov	x2, 8
	mov	x0, x19
	mov	x1, x2
	mov	w5, 0
.LEHB5:
	bl	_Z5printIiEvPT_mmNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEb.isra.0
.LEHE5:
	ldr	x0, [sp, 184]
	cmp	x0, x23
	beq	.L91
	ldr	x1, [sp, 200]
	add	x1, x1, 1
	bl	_ZdlPvm
	ldp	x27, x28, [sp, 80]
	.cfi_remember_state
	.cfi_restore 28
	.cfi_restore 27
	b	.L67
.L61:
	.cfi_restore_state
	mov	x0, x23
.LEHB6:
	bl	_ZNKSt5ctypeIcE13_M_widen_initEv
	ldr	x2, [x23]
	adrp	x0, _ZNKSt5ctypeIcE8do_widenEc
	add	x0, x0, :lo12:_ZNKSt5ctypeIcE8do_widenEc
	mov	w1, 10
	ldr	x2, [x2, 48]
	cmp	x2, x0
	beq	.L62
	mov	x0, x23
	blr	x2
.LEHE6:
	and	w1, w0, 255
	b	.L62
.L58:
	.cfi_restore 27
	.cfi_restore 28
	adrp	x22, :got:_ZSt4cout
	adrp	x1, .LC11
	add	x1, x1, :lo12:.LC11
	mov	x2, 17
	ldr	x21, [x22, #:got_lo12:_ZSt4cout]
	add	x25, sp, 160
	mov	x0, x21
.LEHB7:
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	mov	x0, x21
	bl	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.L67:
	movi	v0.4s, 0
	add	x23, sp, 144
	mov	w21, 16960
	mov	x1, x23
	movk	w21, 0xf, lsl 16
	mov	w0, 2
	stp	q0, q0, [sp, 144]
	bl	clock_gettime
	.p2align 3,,7
.L68:
	mov	x1, x19
	mov	x0, x20
	bl	_Z7stepTLBPsPimmPa.constprop.0
	subs	w21, w21, #1
	bne	.L68
	mov	x1, x25
	mov	w0, 2
	bl	clock_gettime
	ldr	d0, [sp, 168]
	mov	x0, 225833675390976
	movk	x0, 0x41cd, lsl 48
	fmov	d8, x0
	scvtf	d0, d0
	fdiv	d0, d0, d8
	bl	__extenddftf2
	str	q0, [sp, 112]
	ldr	x0, [sp, 160]
	bl	__floatditf
	mov	v1.16b, v0.16b
	ldr	q2, [sp, 112]
	mov	v0.16b, v2.16b
	bl	__addtf3
	ldr	d1, [sp, 152]
	str	q0, [sp, 128]
	scvtf	d0, d1
	fdiv	d0, d0, d8
	bl	__extenddftf2
	str	q0, [sp, 112]
	ldr	x0, [sp, 144]
	bl	__floatditf
	mov	v1.16b, v0.16b
	ldr	q4, [sp, 112]
	mov	v0.16b, v4.16b
	bl	__addtf3
	mov	v1.16b, v0.16b
	ldr	q2, [sp, 128]
	mov	v0.16b, v2.16b
	bl	__subtf3
	bl	__trunctfdf2
	fmov	d9, d0
	movi	v1.4s, 0
	mov	x1, x23
	mov	w0, 2
	stp	q1, q1, [sp, 144]
	bl	clock_gettime
	mov	w9, 16960
	movk	w9, 0xf, lsl 16
	.p2align 3,,7
.L70:
	mov	x1, x20
	mov	x2, 0
	.p2align 3,,7
.L69:
	ldrsh	w3, [x1]
	mov	x0, x2
	str	w3, [x19, x2, lsl 2]
	add	x2, x2, 1
	add	x7, x0, 2
	and	x8, x2, 7
	add	x8, x8, 8
	add	x6, x0, 3
	and	x7, x7, 7
	add	x5, x0, 4
	add	x7, x7, 16
	and	x6, x6, 7
	ldrsh	w11, [x1, 2]
	add	x6, x6, 24
	add	x4, x0, 5
	and	x5, x5, 7
	ldrsh	w10, [x1, 4]
	add	x5, x5, 32
	add	x3, x0, 6
	and	x4, x4, 7
	str	w11, [x19, x8, lsl 2]
	add	x4, x4, 40
	ldrsh	w8, [x1, 6]
	and	x3, x3, 7
	add	x0, x0, 7
	str	w10, [x19, x7, lsl 2]
	add	x3, x3, 48
	ldrsh	w7, [x1, 8]
	and	x0, x0, 7
	str	w8, [x19, x6, lsl 2]
	add	x0, x0, 56
	ldrsh	w6, [x1, 10]
	str	w7, [x19, x5, lsl 2]
	add	x1, x1, 16
	ldrsh	w5, [x1, -4]
	str	w6, [x19, x4, lsl 2]
	ldrsh	w4, [x1, -2]
	str	w5, [x19, x3, lsl 2]
	str	w4, [x19, x0, lsl 2]
	cmp	x2, 8
	bne	.L69
	subs	w9, w9, #1
	bne	.L70
	mov	x1, x25
	mov	w0, 2
	bl	clock_gettime
	adrp	x20, .LC13
	ldr	d0, [sp, 168]
	mov	x0, 225833675390976
	movk	x0, 0x41cd, lsl 48
	fmov	d8, x0
	add	x20, x20, :lo12:.LC13
	scvtf	d0, d0
	fdiv	d0, d0, d8
	bl	__extenddftf2
	str	q0, [sp, 112]
	ldr	x0, [sp, 160]
	bl	__floatditf
	mov	v1.16b, v0.16b
	ldr	q2, [sp, 112]
	mov	v0.16b, v2.16b
	bl	__addtf3
	ldr	d1, [sp, 152]
	str	q0, [sp, 128]
	scvtf	d0, d1
	fdiv	d0, d0, d8
	bl	__extenddftf2
	str	q0, [sp, 112]
	ldr	x0, [sp, 144]
	bl	__floatditf
	mov	v1.16b, v0.16b
	ldr	q4, [sp, 112]
	mov	v0.16b, v4.16b
	bl	__addtf3
	ldr	q2, [sp, 128]
	mov	v1.16b, v0.16b
	mov	v0.16b, v2.16b
	bl	__subtf3
	bl	__trunctfdf2
	fmov	d8, d0
	ldr	x22, [x22, #:got_lo12:_ZSt4cout]
	mov	x2, 18
	adrp	x1, .LC12
	add	x1, x1, :lo12:.LC12
	mov	x0, x22
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	mov	x1, 16960
	mov	x0, x22
	movk	x1, 0xf, lsl 16
	bl	_ZNSo9_M_insertImEERSoT_
	mov	x1, x20
	mov	x19, x0
	mov	x2, 13
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	fmov	d0, d8
	mov	x0, x19
	adrp	x19, .LC14
	add	x19, x19, :lo12:.LC14
	bl	_ZNSo9_M_insertIdEERSoT_
	mov	x21, x0
	mov	x1, x19
	mov	x2, 9
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	mov	x0, x21
	bl	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
	mov	x2, 18
	mov	x0, x22
	adrp	x1, .LC15
	add	x1, x1, :lo12:.LC15
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	mov	x1, 16960
	mov	x0, x22
	movk	x1, 0xf, lsl 16
	bl	_ZNSo9_M_insertImEERSoT_
	mov	x1, x20
	mov	x2, 13
	mov	x20, x0
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	fmov	d0, d9
	mov	x0, x20
	bl	_ZNSo9_M_insertIdEERSoT_
	mov	x1, x19
	mov	x2, 9
	mov	x19, x0
	bl	_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l
	mov	x0, x19
	bl	_ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_.isra.0
.LEHE7:
	adrp	x0, :got:__stack_chk_guard
	ldr	x0, [x0, #:got_lo12:__stack_chk_guard]
	ldr	x2, [sp, 216]
	ldr	x1, [x0]
	subs	x2, x2, x1
	mov	x1, 0
	bne	.L96
	mov	w0, 0
	ldp	x19, x20, [sp, 16]
	ldp	x21, x22, [sp, 32]
	ldp	x23, x24, [sp, 48]
	ldp	x25, x26, [sp, 64]
	ldp	d8, d9, [sp, 96]
	ldp	x29, x30, [sp], 224
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 25
	.cfi_restore 26
	.cfi_restore 23
	.cfi_restore 24
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_restore 72
	.cfi_restore 73
	.cfi_def_cfa_offset 0
	ret
.L91:
	.cfi_def_cfa_offset 224
	.cfi_offset 19, -208
	.cfi_offset 20, -200
	.cfi_offset 21, -192
	.cfi_offset 22, -184
	.cfi_offset 23, -176
	.cfi_offset 24, -168
	.cfi_offset 25, -160
	.cfi_offset 26, -152
	.cfi_offset 27, -144
	.cfi_offset 28, -136
	.cfi_offset 29, -224
	.cfi_offset 30, -216
	.cfi_offset 72, -128
	.cfi_offset 73, -120
	ldp	x27, x28, [sp, 80]
	.cfi_remember_state
	.cfi_restore 28
	.cfi_restore 27
	b	.L67
.L95:
	.cfi_restore_state
.LEHB8:
	bl	_ZSt16__throw_bad_castv
.LEHE8:
.L96:
	.cfi_restore 27
	.cfi_restore 28
	stp	x27, x28, [sp, 80]
	.cfi_offset 28, -136
	.cfi_offset 27, -144
	bl	__stack_chk_fail
.L79:
	ldr	x2, [sp, 184]
	add	x1, sp, 200
	mov	x19, x0
	cmp	x2, x1
	beq	.L76
.L92:
	ldr	x1, [sp, 200]
	mov	x0, x2
	add	x1, x1, 1
	bl	_ZdlPvm
.L76:
	mov	x0, x19
.LEHB9:
	bl	_Unwind_Resume
.LEHE9:
.L80:
.L94:
	ldr	x2, [sp, 184]
	mov	x19, x0
	cmp	x2, x23
	bne	.L92
	b	.L76
.L81:
	b	.L94
	.cfi_endproc
.LFE6157:
	.global	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
.LLSDA6157:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE6157-.LLSDACSB6157
.LLSDACSB6157:
	.uleb128 .LEHB0-.LFB6157
	.uleb128 .LEHE0-.LEHB0
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB1-.LFB6157
	.uleb128 .LEHE1-.LEHB1
	.uleb128 .L79-.LFB6157
	.uleb128 0
	.uleb128 .LEHB2-.LFB6157
	.uleb128 .LEHE2-.LEHB2
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB3-.LFB6157
	.uleb128 .LEHE3-.LEHB3
	.uleb128 .L80-.LFB6157
	.uleb128 0
	.uleb128 .LEHB4-.LFB6157
	.uleb128 .LEHE4-.LEHB4
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB5-.LFB6157
	.uleb128 .LEHE5-.LEHB5
	.uleb128 .L81-.LFB6157
	.uleb128 0
	.uleb128 .LEHB6-.LFB6157
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L79-.LFB6157
	.uleb128 0
	.uleb128 .LEHB7-.LFB6157
	.uleb128 .LEHE7-.LEHB7
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB8-.LFB6157
	.uleb128 .LEHE8-.LEHB8
	.uleb128 .L79-.LFB6157
	.uleb128 0
	.uleb128 .LEHB9-.LFB6157
	.uleb128 .LEHE9-.LEHB9
	.uleb128 0
	.uleb128 0
.LLSDACSE6157:
	.section	.text.startup
	.size	main, .-main
	.align	2
	.p2align 4,,11
	.type	_GLOBAL__sub_I__Z7stepTLBPsPimmPa, %function
_GLOBAL__sub_I__Z7stepTLBPsPimmPa:
.LFB6660:
	.cfi_startproc
	stp	x29, x30, [sp, -32]!
	.cfi_def_cfa_offset 32
	.cfi_offset 29, -32
	.cfi_offset 30, -24
	mov	x29, sp
	str	x19, [sp, 16]
	.cfi_offset 19, -16
	adrp	x19, .LANCHOR0
	add	x19, x19, :lo12:.LANCHOR0
	mov	x0, x19
	bl	_ZNSt8ios_base4InitC1Ev
	mov	x1, x19
	adrp	x0, :got:_ZNSt8ios_base4InitD1Ev
	ldr	x19, [sp, 16]
	adrp	x2, __dso_handle
	ldp	x29, x30, [sp], 32
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 19
	.cfi_def_cfa_offset 0
	add	x2, x2, :lo12:__dso_handle
	ldr	x0, [x0, #:got_lo12:_ZNSt8ios_base4InitD1Ev]
	b	__cxa_atexit
	.cfi_endproc
.LFE6660:
	.size	_GLOBAL__sub_I__Z7stepTLBPsPimmPa, .-_GLOBAL__sub_I__Z7stepTLBPsPimmPa
	.section	.init_array,"aw",%init_array
	.align	3
	.xword	_GLOBAL__sub_I__Z7stepTLBPsPimmPa
	.section	.rodata.cst16,"aM",@progbits,16
	.align	4
.LC16:
	.hword	1
	.hword	2
	.hword	3
	.hword	4
	.hword	5
	.hword	6
	.hword	7
	.hword	8
	.align	4
.LC17:
	.hword	9
	.hword	10
	.hword	11
	.hword	12
	.hword	13
	.hword	14
	.hword	15
	.hword	16
	.align	4
.LC18:
	.hword	17
	.hword	18
	.hword	19
	.hword	20
	.hword	21
	.hword	22
	.hword	23
	.hword	24
	.align	4
.LC19:
	.hword	25
	.hword	26
	.hword	27
	.hword	28
	.hword	29
	.hword	30
	.hword	31
	.hword	32
	.align	4
.LC20:
	.hword	33
	.hword	34
	.hword	35
	.hword	36
	.hword	37
	.hword	38
	.hword	39
	.hword	40
	.align	4
.LC21:
	.hword	41
	.hword	42
	.hword	43
	.hword	44
	.hword	45
	.hword	46
	.hword	47
	.hword	48
	.align	4
.LC22:
	.hword	49
	.hword	50
	.hword	51
	.hword	52
	.hword	53
	.hword	54
	.hword	55
	.hword	56
	.align	4
.LC23:
	.hword	57
	.hword	58
	.hword	59
	.hword	60
	.hword	61
	.hword	62
	.hword	63
	.hword	64
	.section	.rodata
	.align	4
.LC0:
	.xword	0
	.xword	0
	.xword	0
	.xword	0
	.xword	0
	.xword	0
	.xword	0
	.xword	0
	.bss
	.align	3
	.set	.LANCHOR0,. + 0
	.type	_ZStL8__ioinit, %object
	.size	_ZStL8__ioinit, 1
_ZStL8__ioinit:
	.zero	1
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align	3
	.type	DW.ref.__gxx_personality_v0, %object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.xword	__gxx_personality_v0
	.hidden	__dso_handle
	.ident	"GCC: (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0"
	.section	.note.GNU-stack,"",@progbits
