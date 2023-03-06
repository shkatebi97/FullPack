struct cpuinfo_mock_file filesystem[] = {
	{
		.path = "/proc/cpuinfo",
		.size = 414,
		.content =
			"Processor\t: ARMv7 Processor rev 1 (v7l)\n"
			"processor\t: 0\n"
			"BogoMIPS\t: 159.25\n"
			"\n"
			"processor\t: 1\n"
			"BogoMIPS\t: 159.25\n"
			"\n"
			"processor\t: 2\n"
			"BogoMIPS\t: 159.25\n"
			"\n"
			"processor\t: 3\n"
			"BogoMIPS\t: 159.25\n"
			"\n"
			"Features\t: swp half thumb fastmult vfp edsp thumbee neon vfpv3 tls vfpv4 \n"
			"CPU implementer\t: 0x41\n"
			"CPU architecture: 7\n"
			"CPU variant\t: 0x0\n"
			"CPU part\t: 0xc05\n"
			"CPU revision\t: 1\n"
			"\n"
			"Hardware\t: QRD MSM8625Q SKUD\n"
			"Revision\t: 0000\n"
			"Serial\t\t: 420369c3c4d34128\n",
	},
	{
		.path = "/system/build.prop",
		.size = 6465,
		.content =
			"# begin build properties\n"
			"# autogenerated by buildinfo.sh\n"
			"ro.build.id=JZO54K\n"
			"ro.build.display.id=JZO54K.I8552ZTANF1\n"
			"ro.build.version.incremental=I8552ZTANF1\n"
			"ro.build.version.sdk=16\n"
			"ro.build.version.codename=REL\n"
			"ro.build.version.release=4.1.2\n"
			"ro.build.date=Mon Jun 16 12:25:15 KST 2014\n"
			"ro.build.date.utc=1402889115\n"
			"ro.build.type=user\n"
			"ro.build.user=se.infra\n"
			"ro.build.host=R0301-05\n"
			"ro.build.tags=release-keys\n"
			"ro.product.model=GT-I8552\n"
			"ro.product.brand=samsung\n"
			"ro.product.name=delos3gzt\n"
			"ro.product.device=delos3geur\n"
			"ro.product.board=MSM8225\n"
			"ro.product.cpu.abi=armeabi-v7a\n"
			"ro.product.cpu.abi2=armeabi\n"
			"ro.product_ship=true\n"
			"ro.product.manufacturer=samsung\n"
			"ro.product.locale.language=en\n"
			"ro.product.locale.region=GB\n"
			"ro.wifi.channels=\n"
			"ro.board.platform=msm7627a\n"
			"# ro.build.product is obsolete; use ro.product.device\n"
			"ro.build.product=delos3geur\n"
			"# Do not try to parse ro.build.description or .fingerprint\n"
			"ro.build.description=delos3gzt-user 4.1.2 JZO54K I8552ZTANF1 release-keys\n"
			"ro.build.fingerprint=samsung/delos3gzt/delos3geur:4.1.2/JZO54K/I8552ZTANF1:user/release-keys\n"
			"ro.build.characteristics=default\n"
			"# Samsung Specific Properties\n"
			"ro.build.PDA=I8552ZTANF1\n"
			"ro.build.hidden_ver=I8552ZTANF1\n"
			"ro.build.changelist=1181821\n"
			"# end build properties\n"
			"#\n"
			"# system.prop for surf\n"
			"#\n"
			"\n"
			"rild.libpath=/system/lib/libril-qc-qmi-1.so\n"
			"rild.libargs=-d /dev/smd0\n"
			"persist.rild.nitz_plmn=\n"
			"persist.rild.nitz_long_ons_0=\n"
			"persist.rild.nitz_long_ons_1=\n"
			"persist.rild.nitz_long_ons_2=\n"
			"persist.rild.nitz_long_ons_3=\n"
			"persist.rild.nitz_short_ons_0=\n"
			"persist.rild.nitz_short_ons_1=\n"
			"persist.rild.nitz_short_ons_2=\n"
			"persist.rild.nitz_short_ons_3=\n"
			"ril.subscription.types=NV,RUIM\n"
			"DEVICE_PROVISIONED=1\n"
			"keyguard.no_require_sim=1\n"
			"debug.sf.hw=1\n"
			"debug.composition.7x27A.type=mdp\n"
			"debug.composition.7x25A.type=mdp\n"
			"debug.composition.8x25.type=dyn\n"
			"debug.hwc.dynThreshold=1.90\n"
			"dalvik.vm.heapsize=24m\n"
			"persist.gralloc.cp.level3=1\n"
			"ro.lcd_brightness=130\n"
			"ro.sf.lcd_density=240\n"
			"\n"
			"# System property ril adb log on\n"
			"persist.radio.adb_log_on=1\n"
			"\n"
			"#\n"
			"# system props for the cne module\n"
			"#\n"
			"persist.cne.UseCne=none\n"
			"persist.cne.bat.range.low.med=30\n"
			"persist.cne.bat.range.med.high=60\n"
			"persist.cne.loc.policy.op=/system/etc/OperatorPolicy.xml\n"
			"persist.cne.loc.policy.user=/system/etc/UserPolicy.xml\n"
			"persist.cne.bwbased.rat.sel=false\n"
			"persist.cne.snsr.based.rat.mgt=false\n"
			"persist.cne.bat.based.rat.mgt=false\n"
			"persist.cne.rat.acq.time.out=30000\n"
			"persist.cne.rat.acq.retry.tout=0\n"
			"persist.cne.fmc.mode=false\n"
			"persist.cne.fmc.init.time.out=30\n"
			"persist.cne.fmc.comm.time.out=130\n"
			"persist.cne.fmc.retry=false\n"
			"persist.cne.feature=0\n"
			"\n"
			"#\n"
			"# system props for the MM modules\n"
			"#\n"
			"media.stagefright.enable-player=true\n"
			"media.stagefright.enable-meta=false\n"
			"media.stagefright.enable-scan=true\n"
			"media.stagefright.enable-http=true\n"
			"media.stagefright.enable-fma2dp=true\n"
			"media.stagefright.enable-aac=true\n"
			"media.stagefright.enable-qcp=true\n"
			"\n"
			"\n"
			"#\n"
			"# system prop for audio post processing\n"
			"#\n"
			"audio.legacy.postproc=true\n"
			"\n"
			"#\n"
			"# system prop for opengles version\n"
			"#\n"
			"ro.opengles.version=131072\n"
			"\n"
			"#\n"
			"# system props for the data modules\n"
			"#\n"
			"ro.use_data_netmgrd=true\n"
			"persist.data.ds_fmc_app.mode=0\n"
			"\n"
			"#\n"
			"# system props for IMS module\n"
			"#\n"
			"persist.ims.regmanager.mode=0\n"
			"\n"
			"#\n"
			"# system prop for requesting Master role in incoming Bluetooth connection.\n"
			"#\n"
			"ro.bluetooth.request.master=true\n"
			"\n"
			"#\n"
			"# system prop for Bluetooth FTP profile\n"
			"#\n"
			"ro.qualcomm.bluetooth.ftp=true\n"
			"\n"
			"#\n"
			"# system prop for Bluetooth SAP profile\n"
			"#\n"
			"ro.qualcomm.bluetooth.sap=true\n"
			"\n"
			"#\n"
			"# system prop for Bluetooth Auto connect for remote initated connections\n"
			"#\n"
			"ro.bluetooth.remote.autoconnect=true\n"
			"\n"
			"#\n"
			"#system property for Bluetooth discoverability timeout in seconds\n"
			"#0: Always discoverable\n"
			"#debug.bt.discoverable_time=0\n"
			"\n"
			"#\n"
			"# System prop to disable strict mode flash on display\n"
			"#\n"
			"persist.sys.strictmode.visual=false\n"
			"\n"
			"#\n"
			"# System prop to enable/disable OMH. Enabled by default\n"
			"#\n"
			"persist.omh.enabled=1\n"
			"\n"
			"#System prop to enable ehrpd capability\n"
			"ro.config.ehrpd=true\n"
			"\n"
			"# System property for cabl\n"
			"ro.qualcomm.cabl=1\n"
			"\n"
			"#\n"
			"#System prop to determine availability of\n"
			"#analog fm path\n"
			"#\n"
			"ro.fm.analogpath.supported=true\n"
			"\n"
			"#\n"
			"#System property for FM transmitter\n"
			"#\n"
			"ro.fm.transmitter=false\n"
			"\n"
			"#\n"
			"#System property for single instance recording\n"
			"#\n"
			"ro.fm.mulinst.recording.support=false\n"
			"\n"
			"#\n"
			"# system props for SD card emulation of emmc partition\n"
			"#\n"
			"ro.emmc.sdcard.partition=18\n"
			"\n"
			"#\n"
			"# system property to enforce Phone Mode view\n"
			"#\n"
			"ro.screen.layout=normal\n"
			"#\n"
			"# Turn off tiled rendering\n"
			"#\n"
			"debug.enabletr=true\n"
			"#\n"
			"#System prop for setting the pixel format\n"
			"#\n"
			"ro.staticwallpaper.pixelformat=RGB_565\n"
			"\n"
			"#\n"
			"#System prop for disabling the meta data mode for encoder\n"
			"#\n"
			"debug.camcorder.disablemeta=0\n"
			"\n"
			"#\n"
			"# Simulate sdcard on /data/media\n"
			"#\n"
			"persist.fuse_sdcard=true\n"
			"\n"
			"#\n"
			"# System prop for using landscape preview layout in camera\n"
			"#\n"
			"debug.camera.landscape=true\n"
			"\n"
			"#\n"
			"# System prop for capping scroll velocity\n"
			"#\n"
			"ro.max.fling_velocity=4000\n"
			"\n"
			"#\n"
			"# System prop for disabling the rendering dirty regions\n"
			"#\n"
			"hwui.render_dirty_regions=false\n"
			"\n"
			"#\n"
			"# System prop for enabling discontinuity for HLS Variant Playlist\n"
			"#\n"
			"httplive.enable.discontinuity=true\n"
			"\n"
			"#\n"
			"# Enable Dynamic sampling to help with cases like mp3 playback.\n"
			"#\n"
			"dev.pm.dyn_samplingrate=1\n"
			"#\n"
			"# On Strider below period is found to improve power numbers for AAC MP4\n"
			"#\n"
			"dev.pm.dyn_sample_period=700000\n"
			"#\n"
			"# System prop to enable DSDS\n"
			"#\n"
			"persist.dsds.enabled=true\n"
			"\n"
			"persist.multisim.config=dsds\n"
			"\n"
			"ro.multi.rild=true\n"
			"\n"
			"ro.kernel.qemu=0\n"
			"\n"
			"#\n"
			"#snapdragon value add features\n"
			"#\n"
			"ro.qc.sdk.camera.facialproc=false\n"
			"ro.qc.sdk.gestures.camera=false\n"
			"#\n"
			"# Enable swaprect feature\n"
			"#\n"
			"debug.sf.swaprect=1\n"
			"\n"
			"#\n"
			"# Keep SIM state on LPM mode\n"
			"#\n"
			"persist.radio.apm_sim_not_pwdn=1\n"
			"#\n"
			"# ADDITIONAL_BUILD_PROPERTIES\n"
			"#\n"
			"persist.radio.apm_sim_not_pwdn=0\n"
			"dalvik.vm.heapstartsize=5m\n"
			"dalvik.vm.heapgrowthlimit=48m\n"
			"dalvik.vm.heapsize=128m\n"
			"ro.vendor.extension_library=/system/lib/libqc-opt.so\n"
			"dalvik.vm.heaputilization=0.25\n"
			"dalvik.vm.heapidealfree=8388608\n"
			"dalvik.vm.heapconcurrentstart=2097152\n"
			"ro.sec.fle.encryption=true\n"
			"ro.config.ringtone=S_Over_the_horizon.ogg\n"
			"ro.config.ringtone_2=02_Fog_on_the_water.ogg\n"
			"ro.config.notification_sound=S_Whistle.ogg\n"
			"ro.config.notification_sound_2=S_On_time.ogg\n"
			"ro.config.alarm_alert=Walk_in_the_forest.ogg\n"
			"ro.config.media_sound=Media_preview_Touch_the_light.ogg\n"
			"ro.error.receiver.default=com.samsung.receiver.error\n"
			"keyguard.no_require_sim=true\n"
			"ro.com.android.dateformat=MM-dd-yyyy\n"
			"ro.carrier=unknown\n"
			"ro.com.google.clientidbase=android-samsung\n"
			"ro.ril.hsxpa=1\n"
			"ro.ril.gprsclass=10\n"
			"ro.adb.qemud=1\n"
			"ro.com.google.gmsversion=4.1_r6\n"
			"net.bt.name=Android\n"
			"dalvik.vm.stack-trace-file=/data/anr/traces.txt\n"
			"\n",
	},
	{
		.path = "/sys/devices/system/cpu/kernel_max",
		.size = 2,
		.content = "3\n",
	},
	{
		.path = "/sys/devices/system/cpu/possible",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/present",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/online",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/offline",
		.size = 4,
		.content = "1-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/affected_cpus",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
		.size = 8,
		.content = "1209600\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq",
		.size = 7,
		.content = "245760\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_transition_latency",
		.size = 6,
		.content = "50000\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/related_cpus",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies",
		.size = 45,
		.content = "245760 320000 480000 700800 1008000 1209600 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors",
		.size = 67,
		.content = "interactive conservative ondemand userspace powersave performance \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq",
		.size = 7,
		.content = "245760\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver",
		.size = 4,
		.content = "msm\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
		.size = 9,
		.content = "ondemand\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq",
		.size = 7,
		.content = "245760\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state",
		.size = 71,
		.content =
			"245760 2541\n"
			"320000 144\n"
			"480000 164\n"
			"700800 394\n"
			"1008000 2542\n"
			"1209600 2131\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/total_trans",
		.size = 4,
		.content = "258\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/cpufreq/stats/trans_table",
		.size = 521,
		.content =
			"   From  :    To\n"
			"         :    245760    320000    480000    700800   1008000   1209600 \n"
			"   245760:         0         0         0        32         0         0 \n"
			"   320000:         7         0         0         1         0         0 \n"
			"   480000:         5         0         0        13         0         1 \n"
			"   700800:        10         2        12         0        34         0 \n"
			"  1008000:         5         2         2         5         0        63 \n"
			"  1209600:         6         4         4         7        43         0 \n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/physical_package_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_siblings_list",
		.size = 4,
		.content = "0-3\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_siblings",
		.size = 2,
		.content = "f\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/core_id",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/thread_siblings_list",
		.size = 2,
		.content = "0\n",
	},
	{
		.path = "/sys/devices/system/cpu/cpu0/topology/thread_siblings",
		.size = 2,
		.content = "1\n",
	},
	{ NULL },
};

#ifdef __ANDROID__
struct cpuinfo_mock_property properties[] = {
	{
		.key = "DEVICE_PROVISIONED",
		.value = "1",
	},
	{
		.key = "ac3.decode",
		.value = "true",
	},
	{
		.key = "af.resampler.quality",
		.value = "255",
	},
	{
		.key = "audio.decoder_override_check",
		.value = "true",
	},
	{
		.key = "audio.legacy.postproc",
		.value = "true",
	},
	{
		.key = "bluetooth.hotoff.state",
		.value = "1",
	},
	{
		.key = "dalvik.vm.heapconcurrentstart",
		.value = "2097152",
	},
	{
		.key = "dalvik.vm.heapgrowthlimit",
		.value = "48m",
	},
	{
		.key = "dalvik.vm.heapidealfree",
		.value = "8388608",
	},
	{
		.key = "dalvik.vm.heapsize",
		.value = "128m",
	},
	{
		.key = "dalvik.vm.heapstartsize",
		.value = "5m",
	},
	{
		.key = "dalvik.vm.heaputilization",
		.value = "0.25",
	},
	{
		.key = "dalvik.vm.stack-trace-file",
		.value = "/data/anr/traces.txt",
	},
	{
		.key = "debug.camcorder.disablemeta",
		.value = "0",
	},
	{
		.key = "debug.camera.landscape",
		.value = "true",
	},
	{
		.key = "debug.composition.7x25A.type",
		.value = "mdp",
	},
	{
		.key = "debug.composition.7x27A.type",
		.value = "mdp",
	},
	{
		.key = "debug.composition.8x25.type",
		.value = "dyn",
	},
	{
		.key = "debug.composition.type",
		.value = "dyn",
	},
	{
		.key = "debug.enabletr",
		.value = "true",
	},
	{
		.key = "debug.hwc.dynThreshold",
		.value = "1.90",
	},
	{
		.key = "debug.sf.hw",
		.value = "1",
	},
	{
		.key = "debug.sf.swaprect",
		.value = "1",
	},
	{
		.key = "dev.bootcomplete",
		.value = "1",
	},
	{
		.key = "dev.kiessupport",
		.value = "TRUE",
	},
	{
		.key = "dev.pm.dyn_sample_period",
		.value = "700000",
	},
	{
		.key = "dev.pm.dyn_samplingrate",
		.value = "1",
	},
	{
		.key = "gsm.current.phone-type",
		.value = "1,1",
	},
	{
		.key = "gsm.network.type",
		.value = "GPRS:1",
	},
	{
		.key = "gsm.network.type_1",
		.value = "GPRS:1",
	},
	{
		.key = "gsm.operator.alpha",
		.value = "",
	},
	{
		.key = "gsm.operator.alpha_1",
		.value = "",
	},
	{
		.key = "gsm.operator.iso-country",
		.value = "us",
	},
	{
		.key = "gsm.operator.iso-country_1",
		.value = "us",
	},
	{
		.key = "gsm.operator.isroaming",
		.value = "false",
	},
	{
		.key = "gsm.operator.isroaming_1",
		.value = "false",
	},
	{
		.key = "gsm.operator.numeric",
		.value = "310260",
	},
	{
		.key = "gsm.operator.numeric_1",
		.value = "310260",
	},
	{
		.key = "gsm.sim.state",
		.value = "ABSENT",
	},
	{
		.key = "gsm.sim.state_1",
		.value = "ABSENT",
	},
	{
		.key = "gsm.version.baseband",
		.value = "I8552ZTANF1",
	},
	{
		.key = "gsm.version.ril-impl",
		.value = "Qualcomm RIL 1.0",
	},
	{
		.key = "gsm.voice.networktype",
		.value = "GPRS:1",
	},
	{
		.key = "httplive.enable.discontinuity",
		.value = "true",
	},
	{
		.key = "hw.cabl.version",
		.value = "1.0.20120512\n",
	},
	{
		.key = "hwui.render_dirty_regions",
		.value = "false",
	},
	{
		.key = "init.svc.BCS-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.DR-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.DTT-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.KIES-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.SMD-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.adbd",
		.value = "running",
	},
	{
		.key = "init.svc.at_distributor",
		.value = "running",
	},
	{
		.key = "init.svc.atfwd",
		.value = "running",
	},
	{
		.key = "init.svc.bccmd",
		.value = "stopped",
	},
	{
		.key = "init.svc.bluetoothd",
		.value = "running",
	},
	{
		.key = "init.svc.bootanim",
		.value = "stopped",
	},
	{
		.key = "init.svc.cnd",
		.value = "running",
	},
	{
		.key = "init.svc.comp-set",
		.value = "stopped",
	},
	{
		.key = "init.svc.dbus",
		.value = "running",
	},
	{
		.key = "init.svc.debuggerd",
		.value = "running",
	},
	{
		.key = "init.svc.diag_uart_log",
		.value = "running",
	},
	{
		.key = "init.svc.drm",
		.value = "running",
	},
	{
		.key = "init.svc.gpu_dcvsd",
		.value = "running",
	},
	{
		.key = "init.svc.griffon",
		.value = "restarting",
	},
	{
		.key = "init.svc.hciattach",
		.value = "running",
	},
	{
		.key = "init.svc.installd",
		.value = "running",
	},
	{
		.key = "init.svc.keystore",
		.value = "running",
	},
	{
		.key = "init.svc.macloader",
		.value = "running",
	},
	{
		.key = "init.svc.media",
		.value = "running",
	},
	{
		.key = "init.svc.mobex-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.mpdecision",
		.value = "running",
	},
	{
		.key = "init.svc.netd",
		.value = "running",
	},
	{
		.key = "init.svc.netmgrd",
		.value = "running",
	},
	{
		.key = "init.svc.p2p_supplicant",
		.value = "running",
	},
	{
		.key = "init.svc.powersnd",
		.value = "stopped",
	},
	{
		.key = "init.svc.ppd",
		.value = "running",
	},
	{
		.key = "init.svc.qcamerasvr",
		.value = "running",
	},
	{
		.key = "init.svc.qcom-c_core-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.qcom-c_main-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.qcom-post-boot",
		.value = "stopped",
	},
	{
		.key = "init.svc.qcom-post-fs",
		.value = "stopped",
	},
	{
		.key = "init.svc.qcom-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.qcom-usb-sh",
		.value = "stopped",
	},
	{
		.key = "init.svc.qmuxd",
		.value = "running",
	},
	{
		.key = "init.svc.qosmgrd",
		.value = "running",
	},
	{
		.key = "init.svc.ril-daemon1",
		.value = "running",
	},
	{
		.key = "init.svc.ril-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.ril-qmi",
		.value = "stopped",
	},
	{
		.key = "init.svc.rmt_storage",
		.value = "running",
	},
	{
		.key = "init.svc.sdcard",
		.value = "running",
	},
	{
		.key = "init.svc.secril-daemon",
		.value = "running",
	},
	{
		.key = "init.svc.sensord",
		.value = "running",
	},
	{
		.key = "init.svc.servicemanager",
		.value = "running",
	},
	{
		.key = "init.svc.surfaceflinger",
		.value = "running",
	},
	{
		.key = "init.svc.synergy_fm_init",
		.value = "running",
	},
	{
		.key = "init.svc.thermald",
		.value = "running",
	},
	{
		.key = "init.svc.ueventd",
		.value = "running",
	},
	{
		.key = "init.svc.version-set",
		.value = "stopped",
	},
	{
		.key = "init.svc.vold",
		.value = "running",
	},
	{
		.key = "init.svc.wiperiface",
		.value = "stopped",
	},
	{
		.key = "init.svc.wpa_supplicant",
		.value = "running",
	},
	{
		.key = "init.svc.zygote",
		.value = "running",
	},
	{
		.key = "keyguard.no_require_sim",
		.value = "true",
	},
	{
		.key = "lpa.decode",
		.value = "true",
	},
	{
		.key = "media.stagefright.enable-aac",
		.value = "true",
	},
	{
		.key = "media.stagefright.enable-fma2dp",
		.value = "true",
	},
	{
		.key = "media.stagefright.enable-http",
		.value = "true",
	},
	{
		.key = "media.stagefright.enable-meta",
		.value = "false",
	},
	{
		.key = "media.stagefright.enable-player",
		.value = "true",
	},
	{
		.key = "media.stagefright.enable-qcp",
		.value = "true",
	},
	{
		.key = "media.stagefright.enable-scan",
		.value = "true",
	},
	{
		.key = "net.bt.name",
		.value = "Android",
	},
	{
		.key = "net.change",
		.value = "net.nw.cache.orderby",
	},
	{
		.key = "net.dns.cache_size",
		.value = "512",
	},
	{
		.key = "net.dns.cache_ttl",
		.value = "600",
	},
	{
		.key = "net.hostname",
		.value = "android-413f71b1560236d3",
	},
	{
		.key = "net.http.getzip",
		.value = "1",
	},
	{
		.key = "net.http.idle_cache.shutdown",
		.value = "true",
	},
	{
		.key = "net.http.idle_cache.size",
		.value = "40",
	},
	{
		.key = "net.http.threads",
		.value = "10",
	},
	{
		.key = "net.mt.init",
		.value = "DONE",
	},
	{
		.key = "net.nw.cache.orderby",
		.value = "weight",
	},
	{
		.key = "net.nw.cache.prioadvstep",
		.value = "86400000",
	},
	{
		.key = "net.nw.cache.weightadvstep",
		.value = "3600000",
	},
	{
		.key = "net.qtaguid_enabled",
		.value = "1",
	},
	{
		.key = "net.rmnet0.dns1",
		.value = "",
	},
	{
		.key = "net.rmnet0.dns2",
		.value = "",
	},
	{
		.key = "net.rmnet0.gw",
		.value = "",
	},
	{
		.key = "net.rmnet1.dns1",
		.value = "",
	},
	{
		.key = "net.rmnet1.dns2",
		.value = "",
	},
	{
		.key = "net.rmnet1.gw",
		.value = "",
	},
	{
		.key = "net.rmnet2.dns1",
		.value = "",
	},
	{
		.key = "net.rmnet2.dns2",
		.value = "",
	},
	{
		.key = "net.rmnet2.gw",
		.value = "",
	},
	{
		.key = "net.tcp.buffersize.default",
		.value = "4096,87380,110208,4096,16384,110208",
	},
	{
		.key = "net.tcp.buffersize.edge",
		.value = "4093,26280,35040,4096,16384,35040",
	},
	{
		.key = "net.tcp.buffersize.evdo_b",
		.value = "4094,87380,262144,4096,16384,262144",
	},
	{
		.key = "net.tcp.buffersize.gprs",
		.value = "4092,8760,11680,4096,8760,11680",
	},
	{
		.key = "net.tcp.buffersize.hsdpa",
		.value = "4094,87380,1220608,4096,16384,1220608",
	},
	{
		.key = "net.tcp.buffersize.hspa",
		.value = "4094,87380,1220608,4096,16384,1220608",
	},
	{
		.key = "net.tcp.buffersize.hsupa",
		.value = "4094,87380,1220608,4096,16384,1220608",
	},
	{
		.key = "net.tcp.buffersize.lte",
		.value = "4094,87380,1220608,4096,16384,1220608",
	},
	{
		.key = "net.tcp.buffersize.umts",
		.value = "4094,87380,110208,4096,16384,110208",
	},
	{
		.key = "net.tcp.buffersize.wifi",
		.value = "524288,1048576,2097152,262144,524288,1048576",
	},
	{
		.key = "net.webkit.cache.maxdeadsize",
		.value = "4194304",
	},
	{
		.key = "net.webkit.cache.mindeadsize",
		.value = "4194304",
	},
	{
		.key = "net.webkit.cache.size",
		.value = "12582912",
	},
	{
		.key = "persist.audio.allsoundmute",
		.value = "0",
	},
	{
		.key = "persist.audio.headsetsysvolume",
		.value = "4",
	},
	{
		.key = "persist.audio.hphonesysvolume",
		.value = "4",
	},
	{
		.key = "persist.audio.ringermode",
		.value = "2",
	},
	{
		.key = "persist.audio.sysvolume",
		.value = "4",
	},
	{
		.key = "persist.cne.UseCne",
		.value = "none",
	},
	{
		.key = "persist.cne.bat.based.rat.mgt",
		.value = "false",
	},
	{
		.key = "persist.cne.bat.range.low.med",
		.value = "30",
	},
	{
		.key = "persist.cne.bat.range.med.high",
		.value = "60",
	},
	{
		.key = "persist.cne.bwbased.rat.sel",
		.value = "false",
	},
	{
		.key = "persist.cne.feature",
		.value = "0",
	},
	{
		.key = "persist.cne.fmc.comm.time.out",
		.value = "130",
	},
	{
		.key = "persist.cne.fmc.init.time.out",
		.value = "30",
	},
	{
		.key = "persist.cne.fmc.mode",
		.value = "false",
	},
	{
		.key = "persist.cne.fmc.retry",
		.value = "false",
	},
	{
		.key = "persist.cne.loc.policy.op",
		.value = "/system/etc/OperatorPolicy.xml",
	},
	{
		.key = "persist.cne.loc.policy.user",
		.value = "/system/etc/UserPolicy.xml",
	},
	{
		.key = "persist.cne.rat.acq.retry.tout",
		.value = "0",
	},
	{
		.key = "persist.cne.rat.acq.time.out",
		.value = "30000",
	},
	{
		.key = "persist.cne.snsr.based.rat.mgt",
		.value = "false",
	},
	{
		.key = "persist.data.ds_fmc_app.mode",
		.value = "0",
	},
	{
		.key = "persist.dsds.enabled",
		.value = "true",
	},
	{
		.key = "persist.fuse_sdcard",
		.value = "true",
	},
	{
		.key = "persist.gralloc.cp.level3",
		.value = "1",
	},
	{
		.key = "persist.ims.regmanager.mode",
		.value = "0",
	},
	{
		.key = "persist.multisim.config",
		.value = "dsds",
	},
	{
		.key = "persist.omh.enabled",
		.value = "1",
	},
	{
		.key = "persist.radio.adb_log_on",
		.value = "1",
	},
	{
		.key = "persist.radio.apm_sim_not_pwdn",
		.value = "0",
	},
	{
		.key = "persist.radio.eons.enabled",
		.value = "true",
	},
	{
		.key = "persist.radio.plmnname_1",
		.value = "",
	},
	{
		.key = "persist.radio.plmnname_2",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_long_ons_0",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_long_ons_1",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_long_ons_2",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_long_ons_3",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_plmn",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_short_ons_0",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_short_ons_1",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_short_ons_2",
		.value = "",
	},
	{
		.key = "persist.rild.nitz_short_ons_3",
		.value = "",
	},
	{
		.key = "persist.sys.country",
		.value = "US",
	},
	{
		.key = "persist.sys.flipfontpath",
		.value = "default",
	},
	{
		.key = "persist.sys.gps_boot",
		.value = "1",
	},
	{
		.key = "persist.sys.language",
		.value = "en",
	},
	{
		.key = "persist.sys.localevar",
		.value = "",
	},
	{
		.key = "persist.sys.profiler_ms",
		.value = "0",
	},
	{
		.key = "persist.sys.setupwizard",
		.value = "FINISH",
	},
	{
		.key = "persist.sys.storage_preload",
		.value = "2",
	},
	{
		.key = "persist.sys.strictmode.visual",
		.value = "false",
	},
	{
		.key = "persist.sys.usb.config",
		.value = "mtp,adb",
	},
	{
		.key = "ril.CardSlotStatus",
		.value = "0",
	},
	{
		.key = "ril.ICC_TYPE",
		.value = "0",
	},
	{
		.key = "ril.ICC_TYPE_1",
		.value = "0",
	},
	{
		.key = "ril.MSIMM",
		.value = "0",
	},
	{
		.key = "ril.approved_codever",
		.value = "none",
	},
	{
		.key = "ril.approved_cscver",
		.value = "none",
	},
	{
		.key = "ril.approved_modemver",
		.value = "none",
	},
	{
		.key = "ril.atd_status",
		.value = "1_1_0",
	},
	{
		.key = "ril.deviceOffRes",
		.value = "0",
	},
	{
		.key = "ril.ecclist0_0",
		.value = "112,911,999,000,110,118,119",
	},
	{
		.key = "ril.ecclist",
		.value = "911,112,000,08,110,999,118,119",
	},
	{
		.key = "ril.eri_num",
		.value = "0",
	},
	{
		.key = "ril.hw_ver",
		.value = "MP 1.200",
	},
	{
		.key = "ril.isSimChanged",
		.value = "0",
	},
	{
		.key = "ril.isSimChanged_1",
		.value = "0",
	},
	{
		.key = "ril.model_id",
		.value = "",
	},
	{
		.key = "ril.modem.board",
		.value = "MSM8225Q",
	},
	{
		.key = "ril.modem_sim_status",
		.value = "1",
	},
	{
		.key = "ril.official_cscver",
		.value = "I8552ZZTANF1",
	},
	{
		.key = "ril.prl_num",
		.value = "0",
	},
	{
		.key = "ril.product_code",
		.value = "GT-I8552TAABRI",
	},
	{
		.key = "ril.rfcal_date",
		.value = "2014.07.03",
	},
	{
		.key = "ril.sales_code",
		.value = "BRI",
	},
	{
		.key = "ril.serialnumber",
		.value = "RV1F62HGAYZ",
	},
	{
		.key = "ril.subscription.types",
		.value = "NV,RUIM",
	},
	{
		.key = "ril.sw_ver",
		.value = "I8552ZTANF1",
	},
	{
		.key = "rild.libargs",
		.value = "-d /dev/smd0",
	},
	{
		.key = "rild.libpath",
		.value = "/system/lib/libril-qc-qmi-1.so",
	},
	{
		.key = "ro.adb.qemud",
		.value = "1",
	},
	{
		.key = "ro.allow.mock.location",
		.value = "0",
	},
	{
		.key = "ro.baseband",
		.value = "msm",
	},
	{
		.key = "ro.bluetooth.remote.autoconnect",
		.value = "true",
	},
	{
		.key = "ro.bluetooth.request.master",
		.value = "true",
	},
	{
		.key = "ro.board.platform",
		.value = "msm7627a",
	},
	{
		.key = "ro.boot.baseband",
		.value = "msm",
	},
	{
		.key = "ro.boot.bootloader",
		.value = "I8552ZCANG1",
	},
	{
		.key = "ro.boot.debug_level",
		.value = "0x4f4c",
	},
	{
		.key = "ro.boot.emmc",
		.value = "true",
	},
	{
		.key = "ro.boot.emmc_checksum",
		.value = "3",
	},
	{
		.key = "ro.boot.hardware",
		.value = "qcom",
	},
	{
		.key = "ro.boot.serialno",
		.value = "69c3c4d3",
	},
	{
		.key = "ro.boot_recovery",
		.value = "unknown",
	},
	{
		.key = "ro.bootchg",
		.value = "unknown",
	},
	{
		.key = "ro.bootloader",
		.value = "I8552ZCANG1",
	},
	{
		.key = "ro.bootmode",
		.value = "unknown",
	},
	{
		.key = "ro.bt.bdaddr_path",
		.value = "/efs/bluetooth/bt_addr",
	},
	{
		.key = "ro.build.PDA",
		.value = "I8552ZTANF1",
	},
	{
		.key = "ro.build.changelist",
		.value = "1181821",
	},
	{
		.key = "ro.build.characteristics",
		.value = "default",
	},
	{
		.key = "ro.build.date.utc",
		.value = "1402889115",
	},
	{
		.key = "ro.build.date",
		.value = "Mon Jun 16 12:25:15 KST 2014",
	},
	{
		.key = "ro.build.description",
		.value = "delos3gzt-user 4.1.2 JZO54K I8552ZTANF1 release-keys",
	},
	{
		.key = "ro.build.display.id",
		.value = "JZO54K.I8552ZTANF1",
	},
	{
		.key = "ro.build.fingerprint",
		.value = "samsung/delos3gzt/delos3geur:4.1.2/JZO54K/I8552ZTANF1:user/release-keys",
	},
	{
		.key = "ro.build.hidden_ver",
		.value = "I8552ZTANF1",
	},
	{
		.key = "ro.build.host",
		.value = "R0301-05",
	},
	{
		.key = "ro.build.id",
		.value = "JZO54K",
	},
	{
		.key = "ro.build.product",
		.value = "delos3geur",
	},
	{
		.key = "ro.build.tags",
		.value = "release-keys",
	},
	{
		.key = "ro.build.type",
		.value = "user",
	},
	{
		.key = "ro.build.user",
		.value = "se.infra",
	},
	{
		.key = "ro.build.version.codename",
		.value = "REL",
	},
	{
		.key = "ro.build.version.incremental",
		.value = "I8552ZTANF1",
	},
	{
		.key = "ro.build.version.release",
		.value = "4.1.2",
	},
	{
		.key = "ro.build.version.sdk",
		.value = "16",
	},
	{
		.key = "ro.carrier",
		.value = "unknown",
	},
	{
		.key = "ro.com.android.dateformat",
		.value = "MM-dd-yyyy",
	},
	{
		.key = "ro.com.google.clientidbase",
		.value = "android-samsung",
	},
	{
		.key = "ro.com.google.gmsversion",
		.value = "4.1_r6",
	},
	{
		.key = "ro.config.alarm_alert",
		.value = "Walk_in_the_forest.ogg",
	},
	{
		.key = "ro.config.ehrpd",
		.value = "true",
	},
	{
		.key = "ro.config.media_sound",
		.value = "Media_preview_Touch_the_light.ogg",
	},
	{
		.key = "ro.config.notification_sound",
		.value = "S_Whistle.ogg",
	},
	{
		.key = "ro.config.notification_sound_2",
		.value = "S_On_time.ogg",
	},
	{
		.key = "ro.config.ringtone",
		.value = "S_Over_the_horizon.ogg",
	},
	{
		.key = "ro.config.ringtone_2",
		.value = "02_Fog_on_the_water.ogg",
	},
	{
		.key = "ro.cp_debug_level",
		.value = "unknown",
	},
	{
		.key = "ro.crypto.state",
		.value = "unencrypted",
	},
	{
		.key = "ro.csc.country_code",
		.value = "Taiwan",
	},
	{
		.key = "ro.csc.countryiso_code",
		.value = "TW",
	},
	{
		.key = "ro.csc.sales_code",
		.value = "BRI",
	},
	{
		.key = "ro.debug_level",
		.value = "0x4f4c",
	},
	{
		.key = "ro.debuggable",
		.value = "0",
	},
	{
		.key = "ro.emmc.sdcard.partition",
		.value = "18",
	},
	{
		.key = "ro.emmc",
		.value = "true",
	},
	{
		.key = "ro.emmc_checksum",
		.value = "3",
	},
	{
		.key = "ro.error.receiver.default",
		.value = "com.samsung.receiver.error",
	},
	{
		.key = "ro.factorytest",
		.value = "0",
	},
	{
		.key = "ro.fm.analogpath.supported",
		.value = "true",
	},
	{
		.key = "ro.fm.mulinst.recording.support",
		.value = "false",
	},
	{
		.key = "ro.fm.transmitter",
		.value = "false",
	},
	{
		.key = "ro.fuse_sdcard",
		.value = "true",
	},
	{
		.key = "ro.hardware",
		.value = "qcom",
	},
	{
		.key = "ro.hw_plat",
		.value = "8x25",
	},
	{
		.key = "ro.kernel.qemu",
		.value = "0",
	},
	{
		.key = "ro.lcd_brightness",
		.value = "130",
	},
	{
		.key = "ro.max.fling_velocity",
		.value = "4000",
	},
	{
		.key = "ro.multi.rild",
		.value = "true",
	},
	{
		.key = "ro.nvdata_backup",
		.value = "unknown",
	},
	{
		.key = "ro.opengles.version",
		.value = "131072",
	},
	{
		.key = "ro.product.board",
		.value = "MSM8225",
	},
	{
		.key = "ro.product.brand",
		.value = "samsung",
	},
	{
		.key = "ro.product.cpu.abi2",
		.value = "armeabi",
	},
	{
		.key = "ro.product.cpu.abi",
		.value = "armeabi-v7a",
	},
	{
		.key = "ro.product.device",
		.value = "delos3geur",
	},
	{
		.key = "ro.product.locale.language",
		.value = "en",
	},
	{
		.key = "ro.product.locale.region",
		.value = "GB",
	},
	{
		.key = "ro.product.manufacturer",
		.value = "samsung",
	},
	{
		.key = "ro.product.model",
		.value = "GT-I8552",
	},
	{
		.key = "ro.product.name",
		.value = "delos3gzt",
	},
	{
		.key = "ro.product_ship",
		.value = "true",
	},
	{
		.key = "ro.qc.sdk.audio.fluencetype",
		.value = "fluence",
	},
	{
		.key = "ro.qc.sdk.camera.facialproc",
		.value = "false",
	},
	{
		.key = "ro.qc.sdk.gestures.camera",
		.value = "false",
	},
	{
		.key = "ro.qualcomm.bluetooth.ftp",
		.value = "true",
	},
	{
		.key = "ro.qualcomm.bluetooth.sap",
		.value = "true",
	},
	{
		.key = "ro.qualcomm.cabl",
		.value = "1",
	},
	{
		.key = "ro.revision",
		.value = "0",
	},
	{
		.key = "ro.ril.gprsclass",
		.value = "10",
	},
	{
		.key = "ro.ril.hsxpa",
		.value = "1",
	},
	{
		.key = "ro.runtime.firstboot",
		.value = "1325636102182",
	},
	{
		.key = "ro.screen.layout",
		.value = "normal",
	},
	{
		.key = "ro.sec.fle.encryption",
		.value = "true",
	},
	{
		.key = "ro.secure",
		.value = "1",
	},
	{
		.key = "ro.serialno",
		.value = "69c3c4d3",
	},
	{
		.key = "ro.setupwizard.mode",
		.value = "OPTIONAL",
	},
	{
		.key = "ro.sf.lcd_density",
		.value = "240",
	},
	{
		.key = "ro.staticwallpaper.pixelformat",
		.value = "RGB_565",
	},
	{
		.key = "ro.telephony.call_ring.multiple",
		.value = "false",
	},
	{
		.key = "ro.use_data_netmgrd",
		.value = "true",
	},
	{
		.key = "ro.vendor.extension_library",
		.value = "/system/lib/libqc-opt.so",
	},
	{
		.key = "ro.wifi.channels",
		.value = "",
	},
	{
		.key = "service.bootanim.exit",
		.value = "1",
	},
	{
		.key = "service.media.powersnd",
		.value = "1",
	},
	{
		.key = "sys.boot_completed",
		.value = "1",
	},
	{
		.key = "sys.media.vdec.sw",
		.value = "0",
	},
	{
		.key = "sys.settings_system_version",
		.value = "17",
	},
	{
		.key = "sys.usb.config",
		.value = "mtp,adb",
	},
	{
		.key = "sys.usb.state",
		.value = "mtp,adb",
	},
	{
		.key = "system_init.startsurfaceflinger",
		.value = "0",
	},
	{
		.key = "use.non-omx.aac.decoder",
		.value = "true",
	},
	{
		.key = "use.non-omx.mp3.decoder",
		.value = "true",
	},
	{
		.key = "vold.post_fs_data_done",
		.value = "1",
	},
	{
		.key = "wifi.interface",
		.value = "wlan0",
	},
	{
		.key = "wlan.driver.status",
		.value = "ok",
	},
	{ NULL },
};
#endif /* __ANDROID__ */
