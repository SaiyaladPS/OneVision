const STRINGS = {
  th: {
    image: "รูปภาพ",
    video: "วิดีโอ",
    camera: "กล้อง",
    mode_image: "สแกนจากรูปภาพ",
    mode_video: "สแกนจากวิดีโอ",
    mode_camera: "กล้องวงจรปิด",
    hint_image: "เปิดรูป แล้วลากกรอบบนจุดที่ป้ายปรากฏ — สแกนเฉพาะในกรอบ",
    hint_video: "เปิดไฟล์วิดีโอ แล้วเริ่มสแกนทีละเฟรม",
    hint_camera: "ติ๊กเลือกกล้องเพื่อแสดงภาพและสแกนป้าย — ยกเลิกเพื่อปิดกล้อง",
    cameras_menu: "ตั้งค่ากล้อง",
    cameras_close: "ปิด",
    open_image: "เปิดรูปภาพ",
    open_video: "เปิดวิดีโอ",
    start_scan: "เริ่มสแกน",
    stop_scan: "หยุดสแกน",
    save_json: "บันทึก JSON",
    no_file: "ยังไม่ได้เลือกไฟล์",
    roi_enabled: "เปิดใช้กรอบ",
    roi_shape: "รูปทรง",
    shape_rectangle: "สี่เหลี่ยม",
    shape_circle: "วงกลม",
    shape_ellipse: "วงรี",
    roi_full: "เต็มภาพ",
    roi_center: "กึ่งกลาง",
    roi_head: "พื้นที่สแกน (ROI)",
    roi_head_camera: "พื้นที่สแกน · {label}",
    roi_tip: "สแกนเฉพาะในกรอบนี้ · ลากในกรอบเพื่อย้าย · ลากมุมเพื่อย่อขยาย · ลากนอกกรอบเพื่อสร้างใหม่",
    roi_tip_camera: "เลือกกล้อง แล้วลากกรอบบนช่องนั้น — แต่ละกล้องมีพื้นที่สแกนของตัวเอง",
    preview_head: "ภาพและกรอบป้าย",
    crop_head: "ภาพครอปก่อน OCR",
    crop_empty: "ยังไม่มีภาพครอป — เริ่มสแกนแล้วป้ายจะโผล่ที่นี่",
    plate_head: "รูปแบบป้ายทะเบียน",
    detail_head: "รายละเอียดป้ายที่เลือก",
    table_head: "รายการผลตรวจ",
    table_empty: "ยังไม่มีผลตรวจ",
    col_plate: "ป้าย",
    vehicle_car: "รถยนต์",
    vehicle_bus: "รถบัส",
    vehicle_truck: "รถบรรทุก",
    vehicle_van: "รถตู้",
    vehicle_pickup: "กระบะ",
    vehicle_motorcycle: "มอเตอร์ไซค์",
    vehicle_tuk_tuk: "ตุ๊กตุ๊ก",
    vehicle_other: "อื่น ๆ",
    empty: "ลากไฟล์มาวาง หรือเปิดจากปุ่มด้านบน",
    ready: "พร้อมใช้งาน — เลือกโหมดแล้วเปิดไฟล์",
    scanning: "กำลังสแกน...",
    cam_start: "เปิดกล้อง",
    cam_close: "ปิดกล้อง",
    cam_snap: "ถ่ายภาพ",
    cam_record: "เริ่มอัด",
    cam_stop: "หยุดอัด",
    cam_live: "กำลังดู {host}",
    cam_recording: "กำลังอัด {host}",
    cam_snap_ok: "ถ่ายภาพแล้ว — กดเริ่มสแกน",
    cam_record_ok: "อัดคลิปแล้ว — กดเริ่มสแกน",
    cam_open_error: "เปิดกล้องไม่ได้",
    worker_head: "AI Worker",
    worker_hint: "ติ๊กเลือกกล้องที่ต้องการให้แสดงภาพและสแกนป้าย — ยกเลิกเพื่อปิดกล้องนั้น",
    camera_list_head: "รายการกล้อง",
    camera_list_hint: "เลือกกล้องจากรายการนี้เพื่อเปิดหรือปิดการใช้งาน",
    camera_list_selected: "ใช้งาน {selected}/{total}",
    camera_select_all: "เปิดทั้งหมด",
    camera_clear_all: "ปิดทั้งหมด",
    start_all_cameras: "เริ่มสแกนกล้องที่เลือก",
    stop_all_cameras: "ปิดกล้องที่กำลังใช้",
    wall_cameras: "{n} กล้อง",
    wall_page: "หน้า {page}/{pages}",
    wall_prev: "ก่อนหน้า",
    wall_next: "ถัดไป",
    compute_label: "ประมวลผล",
    compute_auto: "อัตโนมัติ",
    compute_gpu: "GPU",
    compute_cpu: "CPU",
    compute_hybrid: "GPU+CPU",
    compute_switching: "กำลังสลับโหมด — ไม่ต้องรีสตาร์ท",
    compute_switched: "ใช้โหมด {mode} แล้ว — ไม่ต้องรีสตาร์ท",
    compute_fallback: "เครื่องนี้ไม่มี GPU — ใช้ CPU",
    gpu_cpu: "ใช้ CPU",
    gpu_ready: "GPU {name}",
    gpu_hybrid: "GPU {name} + CPU",
    gpu_loading: "กำลังโหลดโมเดล",
    camera_idle: "ยังไม่เปิด",
    camera_starting: "กำลังเปิดกล้อง",
    camera_stopping: "กำลังปิดกล้อง",
    camera_enable: "ใช้กล้องนี้",
    camera_disable: "ปิดกล้องนี้",
    camera_selected: "กำลังใช้งาน",
    camera_view_only: "ดูภาพเท่านั้น",
    camera_scan_allowed: "สแกนได้",
    camera_scanning: "ตรวจป้าย",
    camera_scanning_load: "กำลังโหลดโมเดลตรวจป้าย",
    camera_scanning_found: "พบป้ายในเฟรม {n}",
    camera_live: "ถ่ายทอดสด",
    ip_camera: "กล้องด่าน",
    ip_camera_placeholder: "192.168.100.191",
    ip_camera_ready: "ใช้กล้องที่ตั้งในเซิร์ฟเวอร์แล้ว",
    ip_camera_add_label: "เพิ่มกล้อง",
    ip_camera_name_label: "ชื่อกล้อง (ไม่บังคับ)",
    ip_camera_edit_name: "ชื่อกล้องที่เลือก",
    ip_camera_edit_url: "RTSP URL (เว้นว่างเพื่อใช้ค่าเดิม)",
    ip_camera_save: "บันทึกข้อมูลกล้อง",
    ip_camera_saved: "บันทึกข้อมูลกล้อง {host} แล้ว",
    ip_camera_add: "เพิ่ม",
    ip_camera_add_placeholder: "IP เช่น 10.0.75.24 หรือ local",
    ip_camera_added: "เพิ่มกล้อง {host} แล้ว — กดเริ่มสแกน",
    ip_camera_add_error: "เพิ่มกล้องไม่ได้",
    ip_camera_user: "ผู้ใช้กล้อง",
    ip_camera_user_placeholder: "admin",
    ip_camera_password: "รหัสผ่านกล้อง",
    ip_camera_password_placeholder: "รหัสกล้องเครื่องนั้น",
    ip_camera_path: "Path RTSP",
    ip_camera_path_placeholder: "เว้นว่างให้ลองอัตโนมัติ หรือ /Streaming/Channels/101",
    ip_camera_remove: "ลบ",
    camera_storage_database: "บันทึกการตั้งค่าใน PostgreSQL",
    camera_storage_file: "ใช้ไฟล์สำรองสำหรับการตั้งค่า",
    ip_camera_removed: "ลบกล้อง {host} แล้ว",
    local_camera: "กล้องคอมนี้",
    local_camera_n: "กล้องคอม {n}",
    local_camera_add: "กล้องคอมนี้",
    db_ok: "PostgreSQL พร้อม",
    db_off: "ยังไม่ตั้งค่าฐานข้อมูล",
    plates: "{n} ป้าย",
    plate_empty: "ยังไม่มีผลลัพธ์",
    unread: "อ่านทะเบียนไม่ได้",
    country: "ประเทศ",
    vehicle: "ประเภทรถ",
    prefix: "คำนำหน้า",
    number: "เลขทะเบียน",
    province: "จังหวัด / แขวง",
    confidence: "ความมั่นใจ",
    method: "วิธีอ่าน",
    ocr: "ข้อความ OCR",
    plate_type: "ชนิดป้าย",
    province_code: "รหัสจังหวัด",
    source_file: "ไฟล์ต้นทาง",
    media: "ชนิดสื่อ",
    thai: "ไทย",
    lao: "ลาว",
    unknown: "ไม่ทราบ",
    logout: "ออกจากระบบ",
    users: "ผู้ใช้",
    users_head: "จัดการข้อมูลผู้ใช้งาน",
    users_hint: "เพิ่มบัญชีด่าน แก้ชื่อ เปลี่ยนบทบาท ตั้งรหัสใหม่ หรือปิดสิทธิ์ผู้ที่ไม่ได้อยู่เวร",
    users_close: "ปิด",
    user_create: "เพิ่มผู้ใช้",
    user_save: "บันทึกบัญชี",
    user_cancel: "ยกเลิกแก้ไข",
    user_edit: "แก้ไข",
    user_delete: "ลบ",
    user_delete_confirm: "ลบบัญชี {name} หรือไม่",
    col_user: "ชื่อผู้ใช้",
    col_name: "ชื่อที่แสดง",
    col_role: "บทบาท",
    col_status: "สถานะ",
    col_actions: "จัดการ",
    role_admin: "ผู้ดูแลระบบ",
    role_operator: "พนักงานสแกน",
    role_viewer: "ผู้ดูผล",
    role_superuser: "Super User (สิทธิ์ทั้งหมด)",
    status_active: "ใช้งาน",
    status_disabled: "ปิดไว้",
    placeholder_user: "ชื่อผู้ใช้",
    placeholder_name: "ชื่อที่แสดง",
    placeholder_password: "รหัสผ่าน",
    placeholder_password_edit: "เว้นว่างถ้าไม่เปลี่ยนรหัส",
    action_disable: "ปิดสิทธิ์",
    action_enable: "เปิดสิทธิ์",
    no_permission: "บัญชีนี้ไม่มีสิทธิ์รายการนี้",
    signed_in: "{name} · {role}",
    history: "รายการสแกน",
    history_head: "สมุดด่าน — ผลสแกนตามวันและผู้ประจำการ",
    history_hint: "ค้นหาทะเบียน หรือเปิดดูผลตามวันที่ปฏิบัติงานและผู้ประจำการ",
    history_date: "วันที่",
    history_operator: "ผู้ประจำการสแกน",
    history_query: "ค้นหาทะเบียน",
    history_all_operators: "ทุกคน",
    history_close: "ปิดสมุดด่าน",
    history_print: "พิมพ์รายการ",
    history_empty: "วันนี้ยังไม่มีรายการสแกน",
    history_empty_search: "ไม่พบทะเบียนที่ค้นหา",
    history_db_off: "ยังไม่ได้ตั้งค่าฐานข้อมูล จึงยังไม่มีสมุดด่าน",
    history_slip: "ใบตรวจด่าน",
    history_slip_close: "ปิดใบตรวจ",
    history_no_image: "ไม่มีภาพหลักฐานของรายการนี้",
    operator_unknown: "ไม่ระบุผู้สแกน",
    col_time: "เวลา",
    col_photo: "รูปรถ",
    col_plates: "ป้าย",
    col_source: "แหล่งไฟล์",
    shift_count: "{n} คัน",
    vehicle_photo: "รูปรถ",
    plate_crop: "ครอปป้าย",
    report: "รายงาน",
    report_head: "สรุปผลปฏิบัติงานด่าน",
    report_hint: "เลือกช่วงวันที่ แล้วดูจำนวนรถ สัดส่วน และรายการคันที่บันทึกไว้",
    report_from: "ตั้งแต่วันที่",
    report_to: "ถึงวันที่",
    report_operator: "ผู้ประจำการ",
    report_today: "วันนี้",
    report_week: "7 วัน",
    report_month: "เดือนนี้",
    report_close: "ปิดรายงาน",
    report_print: "พิมพ์รายงาน",
    report_csv: "ดาวน์โหลด CSV",
    report_empty: "ไม่มีรายการในช่วงนี้",
    report_db_off: "ยังไม่ได้ตั้งค่าฐานข้อมูล จึงยังไม่มีรายงาน",
    report_range: "ช่วงปฏิบัติงาน",
    report_vehicles: "คัน",
    report_scans: "งานสแกน",
    report_operators: "ผู้ประจำการ",
    report_by_day: "ตามวัน",
    report_by_hour: "ตามชั่วโมง",
    report_by_operator: "ตามผู้ประจำการ",
    report_by_country: "ตามประเทศ",
    report_by_vehicle: "ตามประเภทรถ",
    report_by_province: "ตามจังหวัด / แขวง",
    report_by_media: "ตามชนิดสื่อ",
    report_by_confidence: "ตามความมั่นใจ",
    report_passages: "รายการคันในช่วงนี้",
    report_truncated: "แสดง {n} คันแรก — ดาวน์โหลด CSV เพื่อดูทั้งหมด",
    col_date: "วันที่",
    col_count: "จำนวน",
    col_share: "สัดส่วน",
    conf_high: "สูง",
    conf_medium: "ปานกลาง",
    conf_low: "ต่ำ",
    media_image: "รูปภาพ",
    media_video: "วิดีโอ",
    media_camera: "กล้อง",
    password: "รหัสผ่าน",
    password_head: "เปลี่ยนรหัสผ่านของบัญชีนี้",
    password_current: "รหัสผ่านปัจจุบัน",
    password_new: "รหัสผ่านใหม่",
    password_save: "บันทึกรหัสผ่าน",
    password_close: "ปิด",
    password_ok: "เปลี่ยนรหัสผ่านแล้ว",
  },
  lo: {
    image: "ຮູບພາບ",
    video: "ວິດີໂອ",
    camera: "ກ້ອງ",
    mode_image: "ສະແກນຈາກຮູບພາບ",
    mode_video: "ສະແກນຈາກວິດີໂອ",
    mode_camera: "ກ້ອງວົງຈອນປິດ",
    hint_image: "ເປີດຮູບ ແລ້ວລາກກອບບ່ອນປ້າຍປະກົດ — ສະແກນເສດໃນກອບ",
    hint_video: "ເປີດໄຟລວິດີໂອ ແລ້ວເລີ່ມສະແກນ",
    hint_camera: "ກາເລືອກກ້ອງເພື່ອສະແດງພາບ ແລະສະແກນປ້າຍ — ຍົກເລີກເພື່ອປິດກ້ອງ",
    cameras_menu: "ຕັ້ງຄ່າກ້ອງ",
    cameras_close: "ປິດ",
    open_image: "ເປີດຮູບພາບ",
    open_video: "ເປີດວິດີໂອ",
    start_scan: "ເລີ່ມສະແກນ",
    stop_scan: "ຢຸດສະແກນ",
    save_json: "ບັນທຶກ JSON",
    no_file: "ຍັງບໍ່ໄດ້ເລືອກໄຟລ",
    roi_enabled: "ເປີດໃຊ້ກອບ",
    roi_shape: "ຮູບຮ່າງ",
    shape_rectangle: "ສີ່ຫຼ່ຽມ",
    shape_circle: "ວົງມົນ",
    shape_ellipse: "ວົງລີ",
    roi_full: "ເຕັມພາບ",
    roi_center: "ກາງພາບ",
    roi_head: "ພື້ນທີ່ສະແກນ (ROI)",
    roi_head_camera: "ພື້ນທີ່ສະແກນ · {label}",
    roi_tip: "ສະແກນເສດໃນກອບນີ້ · ລາກໃນກອບເພື່ອຍ້າຍ · ລາກມຸມເພື່ອປັບຂະໜາດ",
    roi_tip_camera: "ເລືອກກ້ອງ ແລ້ວລາກກອບບนຊ່ອງນັ້ນ — ແຕ່ລະກ້ອງມີພື້ນທີ່ສະແກນຂອງຕົວເອງ",
    preview_head: "ພາບແລະກອບປ້າຍ",
    crop_head: "ພາບຄຣອບກ່ອນ OCR",
    crop_empty: "ຍັງບໍ່ມີພາບຄຣອບ — ເລີ່ມສະແກນແລ້ວປ້າຍຈະຂຶ້ນທີ່ນີ້",
    plate_head: "ຮູບແບບປ້າຍທະບຽນ",
    detail_head: "ລາຍລະອຽດປ້າຍທີ່ເລືອກ",
    table_head: "ລາຍການຜົນກວດ",
    table_empty: "ຍັງບໍ່ມີຜົນກວດ",
    col_plate: "ປ້າຍ",
    vehicle_car: "ລົດໃຫຍ່",
    vehicle_bus: "ລົດເມ",
    vehicle_truck: "ລົດບັນທຸກ",
    vehicle_van: "ລົດຕູ້",
    vehicle_pickup: "ກະບະ",
    vehicle_motorcycle: "ລົດຈັກ",
    vehicle_tuk_tuk: "ຕຸກຕຸກ",
    vehicle_other: "ອື່ນ",
    empty: "ລາກໄຟລມາວາງ ຫຼືເປີດຈາກປຸ່ມດ້ານເທິງ",
    ready: "ພ້ອມໃຊ້ງານ — ເລືອກໂໝດແລ້ວເປີດໄຟລ",
    scanning: "ກຳລັງສະແກນ...",
    cam_start: "ເປີດກ້ອງ",
    cam_close: "ປິດກ້ອງ",
    cam_snap: "ຖ່າຍຮູບ",
    cam_record: "ເລີ່ມອັດ",
    cam_stop: "ຢຸດອັດ",
    cam_live: "ກຳລັງເບິ່ງ {host}",
    cam_recording: "ກຳລັງອັດ {host}",
    cam_snap_ok: "ຖ່າຍຮູບແລ້ວ — ກົດເລີ່ມສະແກນ",
    cam_record_ok: "ອັດຄລິບແລ້ວ — ກົດເລີ່ມສະແກນ",
    cam_open_error: "ເປີດກ້ອງບໍ່ໄດ້",
    worker_head: "AI Worker",
    worker_hint: "ກາເລືອກກ້ອງທີ່ຈະໃຫ້ສະແດງພາບ ແລະສະແກນປ້າຍ — ຍົກເລີກເພື່ອປິດກ້ອງນັ້ນ",
    camera_list_head: "ລາຍການກ້ອງ",
    camera_list_hint: "ເລືອກກ້ອງຈາກລາຍການນີ້ເພື່ອເປີດ ຫຼືປິດການໃຊ້ງານ",
    camera_list_selected: "ໃຊ້ງານ {selected}/{total}",
    camera_select_all: "ເປີດທັງໝົດ",
    camera_clear_all: "ປິດທັງໝົດ",
    start_all_cameras: "ເລີ່ມສະແກນກ້ອງທີ່ເລືອກ",
    stop_all_cameras: "ປິດກ້ອງທີ່ກຳລັງໃຊ້",
    wall_cameras: "{n} ກ້ອງ",
    wall_page: "ໜ້າ {page}/{pages}",
    wall_prev: "ກ່ອນໜ້າ",
    wall_next: "ຕໍ່ໄປ",
    compute_label: "ປະມວນຜົນ",
    compute_auto: "ອັດຕະໂນມັດ",
    compute_gpu: "GPU",
    compute_cpu: "CPU",
    compute_hybrid: "GPU+CPU",
    compute_switching: "ກຳລັງສະຫຼັບໂໝດ — ບໍ່ຕ້ອງຣີສະຕາດ",
    compute_switched: "ໃຊ້ໂໝດ {mode} ແລ້ວ — ບໍ່ຕ້ອງຣີສະຕາດ",
    compute_fallback: "ເຄື່ອງນີ້ບໍ່ມີ GPU — ໃຊ້ CPU",
    gpu_cpu: "ໃຊ້ CPU",
    gpu_ready: "GPU {name}",
    gpu_hybrid: "GPU {name} + CPU",
    gpu_loading: "ກຳລັງໂຫຼດໂມເດວ",
    camera_idle: "ຍັງບໍ່ເປີດ",
    camera_starting: "ກຳລັງເປີດກ້ອງ",
    camera_stopping: "ກຳລັງປິດກ້ອງ",
    camera_enable: "ໃຊ້ກ້ອງນີ້",
    camera_disable: "ປິດກ້ອງນີ້",
    camera_selected: "ກຳລັງໃຊ້ງານ",
    camera_view_only: "ເບິ່ງພາບເທົ່ານັ້ນ",
    camera_scan_allowed: "ສະແກນໄດ້",
    camera_scanning: "ກວດປ້າຍ",
    camera_scanning_load: "ກຳລັງໂຫຼດໂມເດວກວດປ້າຍ",
    camera_scanning_found: "ພົບປ້າຍໃນເຟຣມ {n}",
    camera_live: "ຖ່າຍທອດສົດ",
    ip_camera: "ກ້ອງດ່ານ",
    ip_camera_placeholder: "192.168.100.191",
    ip_camera_ready: "ໃຊ້ກ້ອງທີ່ຕັ້ງໃນເຊີບເວີແລ້ວ",
    ip_camera_add_label: "ເພີ່ມກ້ອງ",
    ip_camera_name_label: "ຊື່ກ້ອງ (ບໍ່ບັງຄັບ)",
    ip_camera_edit_name: "ຊື່ກ້ອງທີ່ເລືອກ",
    ip_camera_edit_url: "RTSP URL (ປະໄວ້ວ່າງເພື່ອໃຊ້ຄ່າເກົ່າ)",
    ip_camera_save: "ບັນທຶກຂໍ້ມູນກ້ອງ",
    ip_camera_saved: "ບັນທຶກກ້ອງ {host} ແລ້ວ",
    ip_camera_add: "ເພີ່ມ",
    ip_camera_add_placeholder: "IP ເຊັ່ນ 10.0.75.24 ຫຼື local",
    ip_camera_added: "ເພີ່ມກ້ອງ {host} ແລ້ວ — ກົດເລີ່ມສະແກນ",
    ip_camera_add_error: "ເພີ່ມກ້ອງບໍ່ໄດ້",
    ip_camera_user: "ຜູ້ໃຊ້ກ້ອງ",
    ip_camera_user_placeholder: "admin",
    ip_camera_password: "ລະຫັດຜ່ານກ້ອງ",
    ip_camera_password_placeholder: "ລະຫັດກ້ອງເຄື່ອງນັ້ນ",
    ip_camera_path: "Path RTSP",
    ip_camera_path_placeholder: "ປ່ອຍວ່າງໃຫ້ລອງອັດຕະໂນມັດ ຫຼື /Streaming/Channels/101",
    ip_camera_remove: "ລຶບ",
    camera_storage_database: "ບັນທຶກການຕັ້ງຄ່າໃນ PostgreSQL",
    camera_storage_file: "ໃຊ້ໄຟລ໌ສຳຮອງສຳລັບການຕັ້ງຄ່າ",
    ip_camera_removed: "ລຶບກ້ອງ {host} ແລ້ວ",
    local_camera: "ກ້ອງຄອມນີ້",
    local_camera_n: "ກ້ອງຄອມ {n}",
    local_camera_add: "ກ້ອງຄອມນີ້",
    db_ok: "PostgreSQL ພ້ອມ",
    db_off: "ຍັງບໍ່ຕັ້ງຄ່າຖານຂໍ້ມູນ",
    plates: "{n} ປ້າຍ",
    plate_empty: "ຍັງບໍ່ມີຜົນລັບ",
    unread: "ອ່ານທະບຽນບໍ່ໄດ້",
    country: "ປະເທດ",
    vehicle: "ປະເພດລົດ",
    prefix: "ຄຳນຳໜ້າ",
    number: "ເລກທະບຽນ",
    province: "ແຂວງ / ຈັງຫວັດ",
    confidence: "ຄວາມໝັ້ນໃຈ",
    method: "ວິທີອ່ານ",
    ocr: "ຂໍ້ຄວາມ OCR",
    plate_type: "ປະເພດປ້າຍ",
    province_code: "ລະຫັດແຂວງ",
    source_file: "ໄຟລຕົ້ນທາງ",
    media: "ປະເພດສື່",
    thai: "ໄທ",
    lao: "ລາວ",
    unknown: "ບໍ່ຮູ້",
    logout: "ອອກຈາກລະບົບ",
    users: "ຜູ້ໃຊ້",
    users_head: "ຈັດການຂໍ້ມູນຜູ້ໃຊ້",
    users_hint: "ເພີ່ມບັນຊີດ່ານ ແກ້ຊື່ ປ່ຽນບົດບາດ ຕັ້ງລະຫັດໃໝ່ ຫຼືປິດສິດຜູ້ບໍ່ຢູ່ວຽກ",
    users_close: "ປິດ",
    user_create: "ເພີ່ມຜູ້ໃຊ້",
    user_save: "ບັນທຶກບັນຊີ",
    user_cancel: "ຍົກເລີກແກ້",
    user_edit: "ແກ້ໄຂ",
    user_delete: "ລຶບ",
    user_delete_confirm: "ລຶບບັນຊີ {name} ບໍ່",
    col_user: "ຊື່ຜູ້ໃຊ້",
    col_name: "ຊື່ສະແດງ",
    col_role: "ບົດບາດ",
    col_status: "ສະຖານະ",
    col_actions: "ຈັດການ",
    role_admin: "ຜູ້ດູແລລະບົບ",
    role_operator: "ພະນັກງານສະແກນ",
    role_viewer: "ຜູ້ເບິ່ງຜົນ",
    role_superuser: "Super User (ສິດທັງໝົດ)",
    status_active: "ໃຊ້ງານ",
    status_disabled: "ປິດໄວ້",
    placeholder_user: "ຊື່ຜູ້ໃຊ້",
    placeholder_name: "ຊື່ສະແດງ",
    placeholder_password: "ລະຫັດຜ່ານ",
    placeholder_password_edit: "ປ່ອຍວ່າງຖ້າບໍ່ປ່ຽນລະຫັດ",
    action_disable: "ປິດສິດ",
    action_enable: "ເປີດສິດ",
    no_permission: "ບັນຊີນີ້ບໍ່ມີສິດ",
    signed_in: "{name} · {role}",
    history: "ລາຍການສະແກນ",
    history_head: "ສະໝຸດດ່ານ — ຜົນສະແກນຕາມວັນ ແລະ ຜູ້ປະຈຳການ",
    history_hint: "ຄົ້ນຫາທະບຽນ ຫຼືເບິ່ງຜົນຕາມວັນເຮັດວຽກ ແລະ ຜູ້ປະຈຳການ",
    history_date: "ວັນທີ",
    history_operator: "ຜູ້ປະຈຳການສະແກນ",
    history_query: "ຄົ້ນຫາທະບຽນ",
    history_all_operators: "ທຸກຄົນ",
    history_close: "ປິດສະໝຸດດ່ານ",
    history_print: "ພິມລາຍການ",
    history_empty: "ມື້ນີ້ຍັງບໍ່ມີລາຍການສະແກນ",
    history_empty_search: "ບໍ່ພົບທະບຽນທີ່ຄົ້ນຫາ",
    history_db_off: "ຍັງບໍ່ຕັ້ງຄ່າຖານຂໍ້ມູນ ຈຶ່ງຍັງບໍ່ມີສະໝຸດດ່ານ",
    history_slip: "ໃບກວດດ່ານ",
    history_slip_close: "ປິດໃບກວດ",
    history_no_image: "ບໍ່ມີພາບຫຼັກຖານຂອງລາຍການນີ້",
    operator_unknown: "ບໍ່ລະບຸຜູ້ສະແກນ",
    col_time: "ເວລາ",
    col_photo: "ຮູບລົດ",
    col_plates: "ປ້າຍ",
    col_source: "ແຫຼ່ງໄຟລ",
    shift_count: "{n} ຄັນ",
    vehicle_photo: "ຮູບລົດ",
    plate_crop: "ຄຣອບປ້າຍ",
    report: "ລາຍງານ",
    report_head: "ສະຫຼຸບຜົນປະຕິບັດງານດ່ານ",
    report_hint: "ເລືອກຊ່ວງວັນທີ ແລ້ວເບິ່ງຈຳນວນລົດ ສັດສ່ວນ ແລະລາຍການຄັນທີ່ບັນທຶກໄວ້",
    report_from: "ແຕ່ວັນທີ",
    report_to: "ຮອດວັນທີ",
    report_operator: "ຜູ້ປະຈຳການ",
    report_today: "ມື້ນີ້",
    report_week: "7 ວັນ",
    report_month: "ເດືອນນີ້",
    report_close: "ປິດລາຍງານ",
    report_print: "ພິມລາຍງານ",
    report_csv: "ດາວໂຫຼດ CSV",
    report_empty: "ບໍ່ມີລາຍການໃນຊ່ວງນີ້",
    report_db_off: "ຍັງບໍ່ຕັ້ງຄ່າຖານຂໍ້ມູນ ຈຶ່ງຍັງບໍ່ມີລາຍງານ",
    report_range: "ຊ່ວງປະຕິບັດງານ",
    report_vehicles: "ຄັນ",
    report_scans: "ວຽກສະແກນ",
    report_operators: "ຜູ້ປະຈຳການ",
    report_by_day: "ຕາມວັນ",
    report_by_hour: "ຕາມຊົ່ວໂມງ",
    report_by_operator: "ຕາມຜູ້ປະຈຳການ",
    report_by_country: "ຕາມປະເທດ",
    report_by_vehicle: "ຕາມປະເພດລົດ",
    report_by_province: "ຕາມແຂວງ / ຈັງຫວັດ",
    report_by_media: "ຕາມປະເພດສື່",
    report_by_confidence: "ຕາມຄວາມໝັ້ນໃຈ",
    report_passages: "ລາຍການຄັນໃນຊ່ວງນີ້",
    report_truncated: "ສະແດງ {n} ຄັນທຳອິດ — ດາວໂຫຼດ CSV ເພື່ອເບິ່ງທັງໝົດ",
    col_date: "ວັນທີ",
    col_count: "ຈຳນວນ",
    col_share: "ສັດສ່ວນ",
    conf_high: "ສູງ",
    conf_medium: "ປານກາງ",
    conf_low: "ຕ່ຳ",
    media_image: "ຮູບພາບ",
    media_video: "ວິດີໂອ",
    media_camera: "ກ້ອງ",
    password: "ລະຫັດຜ່ານ",
    password_head: "ປ່ຽນລະຫັດຜ່ານບັນຊີນີ້",
    password_current: "ລະຫັດຜ່ານປັດຈຸບັນ",
    password_new: "ລະຫັດຜ່ານໃໝ່",
    password_save: "ບັນທຶກລະຫັດຜ່ານ",
    password_close: "ປິດ",
    password_ok: "ປ່ຽນລະຫັດຜ່ານແລ້ວ",
  },
  en: {
    image: "Image",
    video: "Video",
    camera: "Camera",
    mode_image: "Scan from image",
    mode_video: "Scan from video",
    mode_camera: "Live CCTV",
    hint_image: "Open a photo, then drag the box over the plate — only that region is scanned",
    hint_video: "Open a video file, then scan sampled frames",
    hint_camera: "Check cameras to show their video and scan plates — uncheck to turn them off",
    cameras_menu: "Cameras",
    cameras_close: "Close",
    open_image: "Open image",
    open_video: "Open video",
    start_scan: "Start scan",
    stop_scan: "Stop scan",
    save_json: "Save JSON",
    no_file: "No file selected",
    roi_enabled: "Enable ROI",
    roi_shape: "Shape",
    shape_rectangle: "Rectangle",
    shape_circle: "Circle",
    shape_ellipse: "Ellipse",
    roi_full: "Full frame",
    roi_center: "Center",
    roi_head: "Scan region (ROI)",
    roi_head_camera: "Scan region · {label}",
    roi_tip: "Only this box is scanned · drag inside to move · drag corners to resize · drag outside to redraw",
    roi_tip_camera: "Select a camera, then drag its box — each camera has its own scan region",
    preview_head: "Vehicle frame and plate boxes",
    crop_head: "OCR-ready crops",
    crop_empty: "No crops yet — plates appear here once a scan starts",
    plate_head: "Plate layout",
    detail_head: "Selected plate",
    table_head: "All detections",
    table_empty: "No detections yet",
    col_plate: "Plate",
    vehicle_car: "Car",
    vehicle_bus: "Bus",
    vehicle_truck: "Truck",
    vehicle_van: "Van",
    vehicle_pickup: "Pickup",
    vehicle_motorcycle: "Motorcycle",
    vehicle_tuk_tuk: "Tuk-tuk",
    vehicle_other: "Other",
    empty: "Drop a file here, or open one from the toolbar",
    ready: "Ready — choose a mode and open a file",
    scanning: "Scanning...",
    cam_start: "Open camera",
    cam_close: "Close camera",
    cam_snap: "Take photo",
    cam_record: "Record",
    cam_stop: "Stop recording",
    cam_live: "Live {host}",
    cam_recording: "Recording {host}",
    cam_snap_ok: "Photo captured — press Start scan",
    cam_record_ok: "Clip saved — press Start scan",
    cam_open_error: "Could not open camera",
    worker_head: "AI Worker",
    worker_hint: "Check the cameras you want to show and scan plates — uncheck one to turn it off",
    camera_list_head: "Camera list",
    camera_list_hint: "Use this separate list to turn cameras on or off",
    camera_list_selected: "Active {selected}/{total}",
    camera_select_all: "Enable all",
    camera_clear_all: "Disable all",
    start_all_cameras: "Scan selected cameras",
    stop_all_cameras: "Stop active cameras",
    wall_cameras: "{n} cameras",
    wall_page: "Page {page}/{pages}",
    wall_prev: "Previous",
    wall_next: "Next",
    compute_label: "Compute",
    compute_auto: "Auto",
    compute_gpu: "GPU",
    compute_cpu: "CPU",
    compute_hybrid: "GPU+CPU",
    compute_switching: "Switching mode — no restart needed",
    compute_switched: "Now using {mode} — no restart needed",
    compute_fallback: "No GPU on this PC — using CPU",
    gpu_cpu: "Using CPU",
    gpu_ready: "GPU {name}",
    gpu_hybrid: "GPU {name} + CPU",
    gpu_loading: "Loading models",
    camera_idle: "Idle",
    camera_starting: "Opening camera",
    camera_stopping: "Closing camera",
    camera_enable: "Use this camera",
    camera_disable: "Turn off camera",
    camera_selected: "Active",
    camera_view_only: "View only",
    camera_scan_allowed: "Scanning allowed",
    camera_scanning: "Reading plates",
    camera_scanning_load: "Loading plate models",
    camera_scanning_found: "Saw {n} plate(s) this frame",
    camera_live: "Live",
    ip_camera: "Gate camera",
    ip_camera_placeholder: "192.168.100.191",
    ip_camera_ready: "Using the camera configured on the server",
    ip_camera_add_label: "Add camera",
    ip_camera_name_label: "Camera name (optional)",
    ip_camera_edit_name: "Selected camera name",
    ip_camera_edit_url: "RTSP URL (leave blank to keep current)",
    ip_camera_save: "Save camera changes",
    ip_camera_saved: "Saved camera {host}",
    ip_camera_add: "Add",
    ip_camera_add_placeholder: "IP such as 10.0.75.24, or local",
    ip_camera_added: "Added camera {host} — press Start scan",
    ip_camera_add_error: "Could not add camera",
    ip_camera_user: "Camera user",
    ip_camera_user_placeholder: "admin",
    ip_camera_password: "Camera password",
    ip_camera_password_placeholder: "That camera's password",
    ip_camera_path: "RTSP path",
    ip_camera_path_placeholder: "Leave blank to auto-try, or /Streaming/Channels/101",
    ip_camera_remove: "Remove",
    camera_storage_database: "Camera settings stored in PostgreSQL",
    camera_storage_file: "Using the settings file fallback",
    ip_camera_removed: "Removed camera {host}",
    local_camera: "This PC camera",
    local_camera_n: "PC camera {n}",
    local_camera_add: "This PC camera",
    db_ok: "PostgreSQL ready",
    db_off: "Database not configured",
    plates: "{n} plates",
    plate_empty: "No result yet",
    unread: "Unreadable",
    country: "Country",
    vehicle: "Vehicle type",
    prefix: "Prefix",
    number: "Number",
    province: "Province",
    confidence: "Confidence",
    method: "Method",
    ocr: "OCR text",
    plate_type: "Plate type",
    province_code: "Province code",
    source_file: "Source file",
    media: "Media",
    thai: "Thai",
    lao: "Lao",
    unknown: "Unknown",
    logout: "Sign out",
    users: "Users",
    users_head: "User accounts",
    users_hint: "Add a gate account, change a name or role, set a new password, or disable someone off duty",
    users_close: "Close",
    user_create: "Add user",
    user_save: "Save account",
    user_cancel: "Cancel edit",
    user_edit: "Edit",
    user_delete: "Delete",
    user_delete_confirm: "Delete account {name}?",
    col_user: "Username",
    col_name: "Display name",
    col_role: "Role",
    col_status: "Status",
    col_actions: "Manage",
    role_admin: "Administrator",
    role_operator: "Scanner",
    role_viewer: "Viewer",
    role_superuser: "Super User (all permissions)",
    status_active: "Active",
    status_disabled: "Disabled",
    placeholder_user: "Username",
    placeholder_name: "Display name",
    placeholder_password: "Password",
    placeholder_password_edit: "Leave blank to keep the current password",
    action_disable: "Disable",
    action_enable: "Enable",
    no_permission: "This account cannot use this action",
    signed_in: "{name} · {role}",
    history: "Scan log",
    history_head: "Gate log — scans by day and operator",
    history_hint: "Look up a plate, or review scans by duty date and operator",
    history_date: "Date",
    history_operator: "Operator on duty",
    history_query: "Search plate",
    history_all_operators: "Everyone",
    history_close: "Close log",
    history_print: "Print list",
    history_empty: "No scans recorded for this date",
    history_empty_search: "No matching plates",
    history_db_off: "Database is not configured, so the gate log is empty",
    history_slip: "Checkpoint slip",
    history_slip_close: "Close slip",
    history_no_image: "No evidence image for this record",
    operator_unknown: "Unassigned operator",
    col_time: "Time",
    col_photo: "Vehicle",
    col_plates: "Plate",
    col_source: "Source file",
    shift_count: "{n} vehicles",
    vehicle_photo: "Vehicle photo",
    plate_crop: "Plate crop",
    report: "Reports",
    report_head: "Gate duty briefing",
    report_hint: "Pick a date range to see vehicle counts, shares, and recorded passages",
    report_from: "From",
    report_to: "To",
    report_operator: "Operator",
    report_today: "Today",
    report_week: "7 days",
    report_month: "This month",
    report_close: "Close report",
    report_print: "Print report",
    report_csv: "Download CSV",
    report_empty: "No records in this range",
    report_db_off: "Database is not configured, so reports are empty",
    report_range: "Duty period",
    report_vehicles: "vehicles",
    report_scans: "scan jobs",
    report_operators: "operators",
    report_by_day: "By day",
    report_by_hour: "By hour",
    report_by_operator: "By operator",
    report_by_country: "By country",
    report_by_vehicle: "By vehicle type",
    report_by_province: "By province",
    report_by_media: "By media",
    report_by_confidence: "By confidence",
    report_passages: "Vehicles in this range",
    report_truncated: "Showing the first {n} vehicles — download CSV for the full list",
    col_date: "Date",
    col_count: "Count",
    col_share: "Share",
    conf_high: "High",
    conf_medium: "Medium",
    conf_low: "Low",
    media_image: "Image",
    media_video: "Video",
    media_camera: "Camera",
    password: "Password",
    password_head: "Change this account's password",
    password_current: "Current password",
    password_new: "New password",
    password_save: "Save password",
    password_close: "Close",
    password_ok: "Password updated",
  },
};

const state = {
  lang: "th",
  mode: "image",
  file: null,
  objectUrl: null,
  image: null,
  roi: { enabled: true, shape: "rectangle", x: 0.05, y: 0.1, width: 0.9, height: 0.8 },
  cameraRois: {},
  roiPushTimers: {},
  drag: null,
  jobId: null,
  source: null,
  plates: [],
  selected: -1,
  result: null,
  stream: null,
  recorder: null,
  chunks: [],
  user: null,
  users: [],
  roleCatalog: [],
  statusCatalog: [],
  editingUserId: null,
  report: null,
  ipCameraConfigured: false,
  ipCameras: [],
  cameraStorage: "file",
  cameraAccessAvailable: false,
  liveOpen: false,
  liveOpening: false,
  liveRecording: false,
  workerSocket: null,
  workerSocketRetry: 0,
  workerSocketTimer: 0,
  workerEventsWanted: false,
  workerEventsSuspended: false,
  liveBlob: "",
  worker: null,
  workerBlobs: {},
  selectedHost: "",
  cameraTransitions: {},
  cameraPage: 0,
  wallLayoutFrame: 0,
};

const $ = (id) => document.getElementById(id);
function setStatus(text) {
  const node = $("statusText");
  if (node) node.textContent = text;
}
const t = (key, values = {}) => {
  const table = STRINGS[state.lang] || STRINGS.th;
  let text = table[key] || STRINGS.th[key] || key;
  Object.entries(values).forEach(([name, value]) => {
    text = text.replaceAll(`{${name}}`, String(value));
  });
  return text;
};

function countryLabel(code) {
  if (code === "thai") return t("thai");
  if (code === "lao") return t("lao");
  return t("unknown");
}

function vehicleLabel(value) {
  const raw = String(value || "").trim();
  if (!raw || raw === "unknown") return t("unknown");
  const key = `vehicle_${raw.toLowerCase().replace(/[^a-z0-9]+/g, "_")}`;
  const labeled = t(key);
  return labeled === key ? raw.replaceAll("_", " ") : labeled;
}

function plateText(plate) {
  if (!plate) return t("plate_empty");
  const prefix = plate.plate_prefix || "";
  const number = plate.plate_number || "";
  if (plate.country === "thai" && prefix && number) return `${prefix}-${number}`;
  if (plate.country === "thai" && !prefix && number) {
    return plate.province ? `${plate.province} ${number}` : number;
  }
  if (prefix || number) return [prefix, number].filter(Boolean).join(" ");
  return plate.ocr?.text || t("unread");
}

function can(permission) {
  return Boolean(state.user?.permissions?.includes(permission));
}

function roleLabel(role) {
  return t(`role_${role}`) || role;
}

function renderRoleCatalog() {
  const select = $("newRole");
  if (!select || !state.roleCatalog.length) return;
  const selected = select.value;
  select.innerHTML = state.roleCatalog
    .map((role) => `<option value="${escapeHtml(role.code)}">${escapeHtml(roleLabel(role.code))}</option>`)
    .join("");
  if (state.roleCatalog.some((role) => role.code === selected)) select.value = selected;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function applyPermissions() {
  const scanOf = { image: "scan.image", video: "scan.video", camera: "scan.camera" };
  document.querySelectorAll(".mode[data-mode]").forEach((button) => {
    const allowed = button.dataset.mode === "camera"
      ? can("scan.camera") || state.cameraAccessAvailable
      : can(scanOf[button.dataset.mode]);
    button.disabled = !allowed;
    button.classList.toggle("hidden", !allowed && !can("users.manage"));
  });
  const canScanMode = can(scanOf[state.mode]);
  $("openLabel").parentElement.classList.toggle("hidden", !canScanMode || state.mode === "camera");
  $("scanBtn").classList.toggle("hidden", !canScanMode);
  $("stopBtn").classList.toggle("hidden", !canScanMode);
  $("saveBtn").classList.toggle("hidden", !can("results.save"));
  $("usersBtn").classList.toggle("hidden", !can("users.manage"));
  if ($("camerasBtn")) $("camerasBtn").classList.toggle("hidden", !can("scan.camera"));
  if ($("camerasToolbarBtn")) {
    $("camerasToolbarBtn").classList.toggle("hidden", state.mode !== "camera" || !can("scan.camera"));
  }
  $("userChip").textContent = state.user
    ? t("signed_in", { name: state.user.display_name, role: roleLabel(state.user.role) })
    : "";
  if (!canScanMode) {
    $("scanBtn").disabled = true;
    $("fileInput").disabled = true;
  } else {
    $("fileInput").disabled = false;
  }
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (response.status === 401) {
    window.location.href = "/login";
    throw new Error("auth");
  }
  return response;
}

function retranslate() {
  document.documentElement.lang = state.lang;
  document.querySelectorAll(".mode[data-mode]").forEach((button) => {
    button.textContent = t(button.dataset.mode);
  });
  $("modeTitle").textContent = t(`mode_${state.mode}`);
  $("modeHint").textContent = t(`hint_${state.mode}`);
  $("openLabel").textContent = state.mode === "video" ? t("open_video") : t("open_image");
  $("scanBtn").textContent = state.mode === "camera" ? t("start_all_cameras") : t("start_scan");
  $("stopBtn").textContent = state.mode === "camera" ? t("stop_all_cameras") : t("stop_scan");
  $("saveBtn").textContent = t("save_json");
  $("roiEnabled").checked = state.roi.enabled;
  $("roiEnabledLabel").textContent = t("roi_enabled");
  $("shapeLabel").textContent = t("roi_shape");
  const shape = $("roiShape");
  shape.options[0].text = t("shape_rectangle");
  shape.options[1].text = t("shape_circle");
  shape.options[2].text = t("shape_ellipse");
  $("roiFull").textContent = t("roi_full");
  $("roiCenter").textContent = t("roi_center");
  $("roiHead").textContent = t("roi_head");
  $("roiTip").textContent = state.mode === "camera" ? t("roi_tip_camera") : t("roi_tip");
  if (typeof syncRoiToolbar === "function") syncRoiToolbar();
  $("previewHead").textContent = state.mode === "camera" ? t("worker_head") : t("preview_head");
  if ($("cameraWallPrev")) $("cameraWallPrev").textContent = t("wall_prev");
  if ($("cameraWallNext")) $("cameraWallNext").textContent = t("wall_next");
  if (state.mode === "camera") {
    renderCameraSelector();
    renderCameraGrid();
    layoutCameraWall();
  }
  $("cropHead").textContent = t("crop_head");
  $("plateHead").textContent = t("plate_head");
  $("detailHead").textContent = t("detail_head");
  $("tableHead").textContent = t("table_head");
  $("emptyHint").textContent = t("empty");
  $("fileChip").textContent = state.file ? state.file.name : t("no_file");
  $("camStart").textContent = state.liveOpen ? t("cam_close") : t("cam_start");
  $("camSnap").textContent = t("cam_snap");
  $("camRecord").textContent = t("cam_record");
  $("camStopRec").textContent = t("cam_stop");
  $("ipCameraLabel").textContent = t("ip_camera");
  $("ipCameraAddLabel").textContent = t("ip_camera_add_label");
  if ($("ipCameraNameLabel")) $("ipCameraNameLabel").textContent = t("ip_camera_name_label");
  if ($("ipCameraEditNameLabel")) $("ipCameraEditNameLabel").textContent = t("ip_camera_edit_name");
  if ($("ipCameraEditUrlLabel")) $("ipCameraEditUrlLabel").textContent = t("ip_camera_edit_url");
  if ($("ipCameraSaveBtn")) $("ipCameraSaveBtn").textContent = t("ip_camera_save");
  $("ipCameraAddBtn").textContent = t("ip_camera_add");
  $("ipCameraRemoveBtn").textContent = t("ip_camera_remove");
  if ($("localCameraBtn")) $("localCameraBtn").textContent = t("local_camera_add");
  $("ipCameraUrl").placeholder = t("ip_camera_add_placeholder");
  if ($("ipCameraUserLabel")) $("ipCameraUserLabel").textContent = t("ip_camera_user");
  if ($("ipCameraUser")) $("ipCameraUser").placeholder = t("ip_camera_user_placeholder");
  if ($("ipCameraPasswordLabel")) $("ipCameraPasswordLabel").textContent = t("ip_camera_password");
  if ($("ipCameraPassword")) $("ipCameraPassword").placeholder = t("ip_camera_password_placeholder");
  if ($("ipCameraPathLabel")) $("ipCameraPathLabel").textContent = t("ip_camera_path");
  if ($("ipCameraPath")) $("ipCameraPath").placeholder = t("ip_camera_path_placeholder");
  if ($("cameraStorageStatus")) {
    $("cameraStorageStatus").textContent = state.cameraStorage === "database"
      ? t("camera_storage_database")
      : t("camera_storage_file");
  }
  if ($("workerHead")) $("workerHead").textContent = t("worker_head");
  if ($("workerHint")) $("workerHint").textContent = t("worker_hint");
  if ($("cameraSelectorHead")) $("cameraSelectorHead").textContent = t("camera_list_head");
  if ($("cameraSelectorHint")) $("cameraSelectorHint").textContent = t("camera_list_hint");
  if ($("cameraSelectAll")) $("cameraSelectAll").textContent = t("camera_select_all");
  if ($("cameraClearAll")) $("cameraClearAll").textContent = t("camera_clear_all");
  if ($("computeLabel")) $("computeLabel").textContent = t("compute_label");
  const computeSelect = $("computeMode");
  if (computeSelect) {
    [...computeSelect.options].forEach((option) => {
      option.textContent = t(`compute_${option.value}`);
    });
  }
  $("plateCount").textContent = t("plates", { n: state.plates.length });
  $("logoutBtn").textContent = t("logout");
  if ($("camerasBtn")) $("camerasBtn").textContent = t("cameras_menu");
  if ($("camerasClose")) $("camerasClose").textContent = t("cameras_close");
  if ($("camerasToolbarBtn")) $("camerasToolbarBtn").textContent = t("cameras_menu");
  $("historyBtn").textContent = t("history");
  $("historyHead").textContent = t("history_head");
  $("historyHint").textContent = t("history_hint");
  $("historyDateLabel").textContent = t("history_date");
  $("historyOperatorLabel").textContent = t("history_operator");
  $("historyQueryLabel").textContent = t("history_query");
  $("historyQuery").placeholder = t("history_query");
  $("historyClose").textContent = t("history_close");
  $("historyPrint").textContent = t("history_print");
  $("historySlipHead").textContent = t("history_slip");
  $("historySlipClose").textContent = t("history_slip_close");
  $("historySlipNoImage").textContent = t("history_no_image");
  $("historySlipVehicleCap").textContent = t("vehicle_photo");
  $("historySlipCropCap").textContent = t("plate_crop");
  $("historyOperator").options[0].text = t("history_all_operators");
  $("reportBtn").textContent = t("report");
  $("reportHead").textContent = t("report_head");
  $("reportHint").textContent = t("report_hint");
  $("reportFromLabel").textContent = t("report_from");
  $("reportToLabel").textContent = t("report_to");
  $("reportOperatorLabel").textContent = t("report_operator");
  $("reportToday").textContent = t("report_today");
  $("reportWeek").textContent = t("report_week");
  $("reportMonth").textContent = t("report_month");
  $("reportClose").textContent = t("report_close");
  $("reportPrint").textContent = t("report_print");
  $("reportCsv").textContent = t("report_csv");
  $("reportOperator").options[0].text = t("history_all_operators");
  $("passwordBtn").textContent = t("password");
  $("passwordHead").textContent = t("password_head");
  $("passwordCurrentLabel").textContent = t("password_current");
  $("passwordNewLabel").textContent = t("password_new");
  $("passwordSave").textContent = t("password_save");
  $("passwordClose").textContent = t("password_close");
  $("usersBtn").textContent = t("users");
  $("usersHead").textContent = t("users_head");
  $("usersHint").textContent = t("users_hint");
  $("usersClose").textContent = t("users_close");
  $("userNameLabel").textContent = t("col_user");
  $("userDisplayLabel").textContent = t("col_name");
  $("userRoleLabel").textContent = t("col_role");
  $("userPasswordLabel").textContent = t("password");
  $("userCreateBtn").textContent = state.editingUserId ? t("user_save") : t("user_create");
  $("userCancelEdit").textContent = t("user_cancel");
  $("newUsername").placeholder = t("placeholder_user");
  $("newDisplayName").placeholder = t("placeholder_name");
  $("newPassword").placeholder = state.editingUserId ? t("placeholder_password_edit") : t("placeholder_password");
  if (state.roleCatalog.length) {
    renderRoleCatalog();
  } else {
    const labels = { operator: "role_operator", viewer: "role_viewer", admin: "role_admin", superuser: "role_superuser" };
    Array.from($("newRole").options).forEach((option) => {
      if (labels[option.value]) option.text = t(labels[option.value]);
    });
  }
  applyPermissions();
  if (state.users.length) renderUsers();
  if (state.report) renderReport(state.report);
  const accept = state.mode === "video" ? "video/*" : "image/*";
  $("fileInput").accept = state.mode === "camera" ? "image/*,video/*" : accept;
  if (!state.jobId) setStatus(t("ready"));
  renderResults();
  draw();
}

function setMode(mode) {
  const needed = { image: "scan.image", video: "scan.video", camera: "scan.camera" }[mode];
  const cameraViewAllowed = mode === "camera" && state.cameraAccessAvailable;
  if (needed && state.user && !can(needed) && !cameraViewAllowed) {
    setStatus(t("no_permission"));
    return;
  }
  state.mode = mode;
  document.querySelector(".app")?.classList.toggle("is-camera-mode", mode === "camera");
  document.querySelectorAll(".mode").forEach((button) => {
    button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
  });
  $("view").classList.toggle("hidden", mode === "camera");
  $("cameraGrid").classList.toggle("hidden", mode !== "camera");
  $("cameraGrid").hidden = mode !== "camera";
  if ($("cameraSelector")) {
    $("cameraSelector").classList.toggle("hidden", mode !== "camera");
    $("cameraSelector").hidden = mode !== "camera";
  }
  $("emptyHint").classList.toggle("hidden", mode === "camera");
  if (mode === "camera") {
    ensureCameraRois();
    if (!state.selectedHost && (state.ipCameras || [])[0]?.host) {
      state.selectedHost = state.ipCameras[0].host;
    }
    startWorkerEvents();
    renderCameraSelector();
    renderCameraGrid();
    syncRoiToolbar();
  }
  retranslate();
  if (!state.jobId) {
    $("scanBtn").disabled = !canStartScan();
    setStatus(t("ready"));
  }
}

function acceptFile(file) {
  if (!file) return;
  const video = file.type.startsWith("video/") || /\.(mp4|avi|mov|mkv|wmv|webm)$/i.test(file.name);
  const permission = video ? "scan.video" : "scan.image";
  if (!can(permission)) {
    setStatus(t("no_permission"));
    return;
  }
  state.file = file;
  state.plates = [];
  state.selected = -1;
  state.result = null;
  state.jobId = null;
  $("scanBtn").disabled = false;
  $("saveBtn").disabled = true;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);
  state.image = null;
  if (!video) {
    const image = new Image();
    image.onload = () => {
      state.image = image;
      draw();
    };
    image.src = state.objectUrl;
    if (state.mode === "video") setMode("image");
  } else if (state.mode === "image") {
    setMode("video");
  }
  $("emptyHint").classList.add("hidden");
  $("fileChip").textContent = file.name;
  setStatus(file.name);
  renderResults();
  draw();
}

function geometry() {
  const canvas = $("view");
  const width = canvas.clientWidth || 640;
  const height = canvas.clientHeight || 360;
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  if (!state.image) {
    return { x: 0, y: 0, w: width, h: height, width, height };
  }
  const scale = Math.min(width / state.image.width, height / state.image.height);
  const w = state.image.width * scale;
  const h = state.image.height * scale;
  return { x: (width - w) / 2, y: (height - h) / 2, w, h, width, height };
}

function draw() {
  const canvas = $("view");
  const ctx = canvas.getContext("2d");
  const g = geometry();
  ctx.fillStyle = "#10161b";
  ctx.fillRect(0, 0, g.width, g.height);
  if (state.image) {
    ctx.drawImage(state.image, g.x, g.y, g.w, g.h);
    $("emptyHint").classList.add("hidden");
  }
  const roi = state.roi;
  const rx = g.x + roi.x * g.w;
  const ry = g.y + roi.y * g.h;
  const rw = roi.width * g.w;
  const rh = roi.height * g.h;
  if ((state.image || state.file) && roi.enabled) {
    ctx.strokeStyle = "#2563EB";
    ctx.lineWidth = 3;
    ctx.fillStyle = "rgba(37, 99, 235, 0.16)";
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.fillRect(rx, ry, rw, rh);
    ctx.fillStyle = "#60A5FA";
    [
      [rx, ry],
      [rx + rw, ry],
      [rx, ry + rh],
      [rx + rw, ry + rh],
    ].forEach(([hx, hy]) => {
      ctx.beginPath();
      ctx.arc(hx, hy, 5, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

function clampRoi(roi) {
  const width = Math.max(0.02, Math.min(1, roi.width));
  const height = Math.max(0.02, Math.min(1, roi.height));
  return {
    ...roi,
    width,
    height,
    x: Math.max(0, Math.min(1 - width, roi.x)),
    y: Math.max(0, Math.min(1 - height, roi.y)),
  };
}

function copyRoi(roi = state.roi) {
  return {
    enabled: roi?.enabled !== false,
    shape: roi?.shape || "rectangle",
    x: Number(roi?.x) || 0,
    y: Number(roi?.y) || 0,
    width: Number(roi?.width) || 1,
    height: Number(roi?.height) || 1,
  };
}

function roiColor(host) {
  const colors = ["#2563EB", "#60A5FA", "#22C55E", "#F59E0B"];
  const index = (state.ipCameras || []).findIndex((item) => item.host === host);
  return colors[(index < 0 ? 0 : index) % colors.length];
}

function cameraLabelFor(host) {
  const live = liveCamera(host);
  const listed = (state.ipCameras || []).find((item) => item.host === host);
  if (listed) return listedCameraLabel(listed);
  if (isLocalCameraRef(host) || isLocalCameraRef(live?.host)) return t("local_camera");
  return live?.label || listed?.label || host || "";
}

function ensureCameraRois() {
  (state.ipCameras || []).forEach((camera) => {
    if (camera.host && !state.cameraRois[camera.host]) {
      state.cameraRois[camera.host] = copyRoi(camera.roi || state.roi);
    }
  });
}

function allCameraRois() {
  ensureCameraRois();
  const rois = {};
  Object.keys(state.cameraRois).forEach((host) => {
    rois[host] = copyRoi(state.cameraRois[host]);
  });
  (state.ipCameras || []).forEach((camera) => {
    rois[camera.host] = copyRoi(roiFor(camera.host));
  });
  return rois;
}

function roiFor(host) {
  return copyRoi((host && state.cameraRois[host]) || state.roi);
}

function syncRoiToolbar() {
  const host = state.mode === "camera" ? selectedCameraHost() : "";
  const roi = host ? roiFor(host) : state.roi;
  if ($("roiEnabled")) $("roiEnabled").checked = roi.enabled !== false;
  if ($("roiShape")) $("roiShape").value = roi.shape || "rectangle";
  if ($("roiHead")) {
    $("roiHead").textContent = host ? t("roi_head_camera", { label: cameraLabelFor(host) }) : t("roi_head");
  }
  if ($("roiTip")) $("roiTip").textContent = state.mode === "camera" ? t("roi_tip_camera") : t("roi_tip");
}

function selectCameraHost(host, options = {}) {
  if (!host) return;
  state.selectedHost = host;
  if ($("ipCameraSelect") && $("ipCameraSelect").value !== host) $("ipCameraSelect").value = host;
  state.roi = roiFor(host);
  syncRoiToolbar();
  if (options.redraw === false) return;
  document.querySelectorAll("#cameraGrid [data-host]").forEach((tile) => {
    tile.classList.toggle("is-selected", tile.dataset.host === host);
  });
}

function setRoiFor(host, roi, options = {}) {
  const push = options.push !== false;
  const paint = options.draw !== false;
  const next = clampRoi({ ...copyRoi(roi), enabled: roi.enabled !== false, shape: roi.shape || "rectangle" });
  if (state.mode === "camera") {
    const target = host || selectedCameraHost();
    if (!target) return;
    state.cameraRois[target] = next;
    if (target === selectedCameraHost()) state.roi = { ...next };
    syncRoiToolbar();
    if (paint) drawCameraRoi(target);
    if (push) schedulePushRoi(target, options.immediate === true);
    return;
  }
  state.roi = next;
  $("roiEnabled").checked = next.enabled;
  if ($("roiShape")) $("roiShape").value = next.shape;
  if (paint) draw();
}

function schedulePushRoi(host, immediate) {
  if (!host) return;
  const key = host;
  window.clearTimeout(state.roiPushTimers[key]);
  const run = () => {
    api("/api/worker/roi", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ host, roi: roiFor(host) }),
    }).catch(() => {});
  };
  if (immediate) run();
  else state.roiPushTimers[key] = window.setTimeout(run, 160);
}

function flushRoiSaves() {
  Object.entries(state.cameraRois || {}).forEach(([host, roi]) => {
    window.clearTimeout(state.roiPushTimers[host]);
    fetch("/api/worker/roi", {
      method: "POST",
      credentials: "same-origin",
      keepalive: true,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ host, roi: copyRoi(roi) }),
    }).catch(() => {});
  });
}

function cameraFrameBox(canvas, img) {
  const width = canvas.width;
  const height = canvas.height;
  const nw = img.naturalWidth || 16;
  const nh = img.naturalHeight || 10;
  const scale = Math.min(width / nw, height / nh) || 1;
  const w = Math.max(1, nw * scale);
  const h = Math.max(1, nh * scale);
  return { x: (width - w) / 2, y: (height - h) / 2, w, h, width, height };
}

function strokeRoiPath(ctx, roi, rx, ry, rw, rh) {
  ctx.beginPath();
  if (roi.shape === "circle") {
    const radius = Math.min(rw, rh) / 2;
    ctx.arc(rx + rw / 2, ry + rh / 2, Math.max(1, radius), 0, Math.PI * 2);
  } else if (roi.shape === "ellipse") {
    ctx.ellipse(rx + rw / 2, ry + rh / 2, Math.max(1, rw / 2), Math.max(1, rh / 2), 0, 0, Math.PI * 2);
  } else {
    ctx.rect(rx, ry, rw, rh);
  }
}

function drawCameraRoi(host) {
  const tile = document.querySelector(`#cameraGrid [data-host="${CSS.escape(host)}"]`);
  if (!tile || tile.classList.contains("is-offpage")) return;
  const canvas = tile.querySelector("canvas.camera-roi");
  const img = tile.querySelector("img");
  if (!canvas || !img) return;
  const width = img.clientWidth || tile.clientWidth;
  const height = img.clientHeight || 0;
  if (width < 8 || height < 8) return;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const roi = roiFor(host);
  if (!roi.enabled) return;
  const g = cameraFrameBox(canvas, img);
  const rx = g.x + roi.x * g.w;
  const ry = g.y + roi.y * g.h;
  const rw = roi.width * g.w;
  const rh = roi.height * g.h;
  const color = roiColor(host);
  ctx.strokeStyle = color;
  ctx.lineWidth = host === selectedCameraHost() ? 3 : 2.5;
  ctx.fillStyle = `${color}24`;
  strokeRoiPath(ctx, roi, rx, ry, rw, rh);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = color;
  [
    [rx, ry],
    [rx + rw, ry],
    [rx, ry + rh],
    [rx + rw, ry + rh],
  ].forEach(([hx, hy]) => {
    ctx.beginPath();
    ctx.arc(hx, hy, 5, 0, Math.PI * 2);
    ctx.fill();
  });
}

function cameraPosFromEvent(event, canvas, img) {
  const rect = canvas.getBoundingClientRect();
  const g = cameraFrameBox(canvas, img);
  const px = ((event.clientX - rect.left) / rect.width) * canvas.width;
  const py = ((event.clientY - rect.top) / rect.height) * canvas.height;
  if (px < g.x || py < g.y || px > g.x + g.w || py > g.y + g.h) return null;
  return { x: (px - g.x) / g.w, y: (py - g.y) / g.h };
}

function bindCameraRoi(canvas, host) {
  if (canvas.dataset.bound === "1") return;
  canvas.dataset.bound = "1";
  canvas.addEventListener("pointerdown", (event) => {
    const img = canvas.parentElement.querySelector("img");
    const p = cameraPosFromEvent(event, canvas, img);
    if (!p) return;
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    selectCameraHost(host);
    const roi = roiFor(host);
    const handle = handleAt(p, roi);
    const inside = p.x >= roi.x && p.x <= roi.x + roi.width && p.y >= roi.y && p.y <= roi.y + roi.height;
    state.drag = {
      host,
      handle,
      start: p,
      roi: { ...roi },
      mode: handle ? "resize" : inside ? "move" : "new",
    };
    setRoiFor(host, { ...roi, enabled: true }, { push: false });
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.drag || state.drag.host !== host) return;
    const img = canvas.parentElement.querySelector("img");
    const p = cameraPosFromEvent(event, canvas, img);
    if (!p) return;
    const { start, roi, mode, handle } = state.drag;
    let next = { ...roi, enabled: true };
    if (mode === "move") {
      next = { ...next, x: roi.x + p.x - start.x, y: roi.y + p.y - start.y };
    } else if (mode === "new") {
      next = {
        ...next,
        x: Math.min(start.x, p.x),
        y: Math.min(start.y, p.y),
        width: Math.abs(p.x - start.x),
        height: Math.abs(p.y - start.y),
      };
    } else {
      const right = roi.x + roi.width;
      const bottom = roi.y + roi.height;
      if (handle.includes("l")) {
        next.x = Math.min(p.x, right - 0.02);
        next.width = right - next.x;
      }
      if (handle.includes("r")) next.width = Math.max(0.02, p.x - roi.x);
      if (handle.includes("t")) {
        next.y = Math.min(p.y, bottom - 0.02);
        next.height = bottom - next.y;
      }
      if (handle.includes("b")) next.height = Math.max(0.02, p.y - roi.y);
    }
    setRoiFor(host, next, { push: false });
  });
  const endDrag = () => {
    if (!state.drag || state.drag.host !== host) return;
    state.drag = null;
    schedulePushRoi(host, true);
  };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
}

function posFromEvent(event) {
  const g = geometry();
  const rect = $("view").getBoundingClientRect();
  const px = event.clientX - rect.left;
  const py = event.clientY - rect.top;
  if (px < g.x || py < g.y || px > g.x + g.w || py > g.y + g.h) return null;
  return { x: (px - g.x) / g.w, y: (py - g.y) / g.h };
}

function handleAt(p, roi = state.roi) {
  const handles = {
    tl: { x: roi.x, y: roi.y },
    tr: { x: roi.x + roi.width, y: roi.y },
    bl: { x: roi.x, y: roi.y + roi.height },
    br: { x: roi.x + roi.width, y: roi.y + roi.height },
  };
  for (const [name, point] of Object.entries(handles)) {
    if (Math.abs(p.x - point.x) < 0.02 && Math.abs(p.y - point.y) < 0.02) return name;
  }
  return "";
}

function bindRoi() {
  const canvas = $("view");
  canvas.addEventListener("mousedown", (event) => {
    if (state.mode === "camera" || state.drag?.host) return;
    const p = posFromEvent(event);
    if (!p) return;
    const handle = handleAt(p);
    const roi = state.roi;
    const inside = p.x >= roi.x && p.x <= roi.x + roi.width && p.y >= roi.y && p.y <= roi.y + roi.height;
    state.drag = { handle, start: p, roi: { ...roi }, mode: handle ? "resize" : inside ? "move" : "new" };
    state.roi.enabled = true;
    $("roiEnabled").checked = true;
  });
  window.addEventListener("mousemove", (event) => {
    if (!state.drag || state.drag.host || state.mode === "camera") return;
    const p = posFromEvent(event);
    if (!p) return;
    const { start, roi, mode, handle } = state.drag;
    if (mode === "move") {
      state.roi = clampRoi({ ...roi, x: roi.x + p.x - start.x, y: roi.y + p.y - start.y });
    } else if (mode === "new") {
      state.roi = clampRoi({
        ...roi,
        x: Math.min(start.x, p.x),
        y: Math.min(start.y, p.y),
        width: Math.abs(p.x - start.x),
        height: Math.abs(p.y - start.y),
      });
    } else {
      let next = { ...roi };
      const right = roi.x + roi.width;
      const bottom = roi.y + roi.height;
      if (handle.includes("l")) {
        next.x = Math.min(p.x, right - 0.02);
        next.width = right - next.x;
      }
      if (handle.includes("r")) next.width = Math.max(0.02, p.x - roi.x);
      if (handle.includes("t")) {
        next.y = Math.min(p.y, bottom - 0.02);
        next.height = bottom - next.y;
      }
      if (handle.includes("b")) next.height = Math.max(0.02, p.y - roi.y);
      state.roi = clampRoi(next);
    }
    draw();
  });
  window.addEventListener("mouseup", () => {
    if (state.drag?.host) return;
    state.drag = null;
  });
}

function renderPlate(plate) {
  const face = $("plateFace");
  face.className = `plate-face ${plate?.country || "unknown"}`;
  $("plateCountry").textContent = plate?.country === "thai" ? "THAILAND" : plate?.country === "lao" ? "LAO P.D.R." : "LICENCE PLATE";
  $("plateNumber").textContent = plateText(plate);
  $("plateProvince").textContent = plate?.province || "";
  const rows = [
    ["vehicle", vehicleLabel(plate?.vehicle_type)],
    ["province", plate?.province || "-"],
    ["confidence", plate ? `${(Number(plate.recognition_confidence || 0) * 100).toFixed(1)}%` : "-"],
    ["method", plate?.ocr?.method || "-"],
    ["ocr", plate?.ocr?.raw_text || plate?.ocr?.text || "-"],
  ];
  $("details").innerHTML = rows.map(([key, value]) => `<dt>${t(key)}</dt><dd>${value}</dd>`).join("");
}

function livePlateKey(plate) {
  if (!plate) return "";
  // Track ids can change when a detector loses a plate for a few frames.
  // The registration is the stable identity for the result list; otherwise
  // the same car is rendered as a new card whenever a new track is created.
  const compact = (value) => String(value || "").trim().toLocaleLowerCase().replace(/[\s\-_.]+/g, "");
  const prefix = compact(plate.plate_prefix);
  const number = compact(plate.plate_number);
  if (prefix || number) {
    const camera = compact(plate.camera_host);
    return `registration:${camera}|${prefix}|${number}`;
  }
  if (plate.camera_host) return `camera:${plate.camera_host}:id:${plate.id || 0}`;
  return `live:${plate.live_result_key || plate.id || ""}`;
}

function plateTime(plate, fallback) {
  const captured = Number(plate?.captured_at || 0);
  if (captured > 0) return captured;
  const id = Number(plate?.id || 0);
  if (id > 0) return id;
  return fallback;
}

function sortPlates(plates) {
  return (plates || [])
    .map((plate, index) => ({ plate, index }))
    .sort((a, b) => {
      const ta = plateTime(a.plate, a.index);
      const tb = plateTime(b.plate, b.index);
      if (ta !== tb) return ta - tb;
      const ha = String(a.plate.camera_host || "");
      const hb = String(b.plate.camera_host || "");
      if (ha !== hb) return ha.localeCompare(hb);
      return Number(a.plate.id || 0) - Number(b.plate.id || 0) || a.index - b.index;
    })
    .map((item) => item.plate);
}

function mergeLivePlates(incoming) {
  const byKey = new Map();
  (state.plates || []).forEach((plate, index) => {
    byKey.set(livePlateKey(plate) || `prev:${index}`, plate);
  });
  (incoming || []).forEach((plate, index) => {
    const key = livePlateKey(plate) || `in:${index}`;
    const prev = byKey.get(key) || {};
    byKey.set(key, {
      ...prev,
      ...plate,
      captured_at: Number(prev.captured_at || plate.captured_at || 0) || Date.now() / 1000,
    });
  });
  state.plates = sortPlates([...byKey.values()]);
}

function scrollPaneTo(scroller, child, axis) {
  if (!scroller || !child) return;
  if (axis === "x") {
    const left = child.offsetLeft - 10;
    const right = left + child.offsetWidth + 10;
    if (left < scroller.scrollLeft) scroller.scrollLeft = left;
    else if (right > scroller.scrollLeft + scroller.clientWidth) {
      scroller.scrollLeft = right - scroller.clientWidth;
    }
    return;
  }
  const top = child.offsetTop - 8;
  const bottom = top + child.offsetHeight + 8;
  if (top < scroller.scrollTop) scroller.scrollTop = top;
  else if (bottom > scroller.scrollTop + scroller.clientHeight) {
    scroller.scrollTop = bottom - scroller.clientHeight;
  }
}

function resultsSignature() {
  return `${state.lang}|` + (state.plates || []).map((plate) => [
    livePlateKey(plate),
    plate.crop_url || "",
    plate.province || "",
    plate.vehicle_type || "",
    Number(plate.recognition_confidence || 0).toFixed(3),
    plate.camera_label || plate.camera_host || "",
  ].join(":")).join("|");
}

function resultTableHeaders() {
  return ["#", t("col_plate"), t("province"), t("country"), t("vehicle"), t("confidence")]
    .map((label) => `<th>${escapeHtml(label)}</th>`)
    .join("");
}

function cropCardHtml(plate, index) {
  const img = plate.crop_url
    ? `<img src="${escapeHtml(plate.crop_url)}" alt="" />`
    : `<span class="crop-placeholder">${escapeHtml(t("unread"))}</span>`;
  const country = plate.country === "thai" || plate.country === "lao" ? plate.country : "unknown";
  const meta = [countryLabel(plate.country), plate.province].filter(Boolean).join(" · ");
  return `<button type="button" class="crop-card ${country}${index === state.selected ? " active" : ""}" data-index="${index}" title="${escapeHtml(plateText(plate))}">${img}<span class="crop-seq">${index + 1}</span><strong>${escapeHtml(plateText(plate))}</strong><span class="crop-meta">${escapeHtml(meta)}</span></button>`;
}

function resultRowHtml(plate, index) {
  const confidence = `${(Number(plate.recognition_confidence || 0) * 100).toFixed(1)}%`;
  const host = plate.camera_label || plate.camera_host || "";
  const extras = [plate.province, host].filter(Boolean).join(" · ");
  const plateCell = extras
    ? `<strong>${escapeHtml(plateText(plate))}</strong><small>${escapeHtml(extras)}</small>`
    : `<strong>${escapeHtml(plateText(plate))}</strong>`;
  return `<tr class="${index === state.selected ? "active" : ""}" data-index="${index}"><td>${index + 1}</td><td class="plate-cell">${plateCell}</td><td>${escapeHtml(plate.province || "-")}</td><td>${escapeHtml(countryLabel(plate.country))}</td><td>${escapeHtml(vehicleLabel(plate.vehicle_type))}</td><td class="conf-cell">${confidence}</td></tr>`;
}

function markSelectedResults() {
  document.querySelectorAll("#crops .crop-card").forEach((card) => {
    card.classList.toggle("active", Number(card.dataset.index) === state.selected);
  });
  document.querySelectorAll("#tableBody tr[data-index]").forEach((row) => {
    row.classList.toggle("active", Number(row.dataset.index) === state.selected);
  });
}

function renderResults() {
  $("plateCount").textContent = t("plates", { n: state.plates.length });
  if ($("cropMeta")) $("cropMeta").textContent = state.plates.length ? t("plates", { n: state.plates.length }) : "";
  if ($("tableMeta")) $("tableMeta").textContent = state.plates.length ? t("plates", { n: state.plates.length }) : "";
  const signature = resultsSignature();
  const listChanged = state._resultsSig !== signature;
  if (!state.plates.length) {
    $("crops").innerHTML = `<p class="crops-empty">${escapeHtml(t("crop_empty"))}</p>`;
    $("tableHeader").innerHTML = resultTableHeaders();
    $("tableBody").innerHTML = `<tr class="empty-row"><td colspan="6">${escapeHtml(t("table_empty"))}</td></tr>`;
    state._resultsSig = signature;
    state._resultsSelected = state.selected;
    renderPlate(null);
    return;
  }
  if (listChanged) {
    $("crops").innerHTML = state.plates.map((plate, index) => cropCardHtml(plate, index)).join("");
    $("tableHeader").innerHTML = resultTableHeaders();
    $("tableBody").innerHTML = state.plates.map((plate, index) => resultRowHtml(plate, index)).join("");
  } else {
    markSelectedResults();
  }
  state._resultsSig = signature;
  state._resultsSelected = state.selected;
  renderPlate(state.plates[state.selected] || null);
  scrollPaneTo($("crops"), document.querySelector("#crops .crop-card.active"), "x");
  scrollPaneTo(document.querySelector(".result-table-panel .table-wrap"), document.querySelector("#tableBody tr.active"), "y");
}

function applyJob(data) {
  if (Array.isArray(data.plates)) {
    const previousKey = livePlateKey(state.plates[state.selected]);
    state.plates = sortPlates(data.plates);
    if (previousKey) {
      const idx = state.plates.findIndex((plate) => livePlateKey(plate) === previousKey);
      state.selected = idx >= 0 ? idx : state.plates.length - 1;
    } else if (state.selected < 0 && state.plates.length) {
      state.selected = 0;
    }
  }
  if (data.result) {
    state.result = data.result;
    state.plates = sortPlates(data.result.plates || state.plates);
    $("saveBtn").disabled = !can("results.save");
    if (data.result.annotated_image) {
      const image = new Image();
      image.onload = () => {
        state.image = image;
        draw();
      };
      image.src = data.result.annotated_image;
    }
  }
  if (data.has_preview && (data.media_type === "video" || data.media_type === "camera")) {
    const image = new Image();
    image.onload = () => {
      state.image = image;
      $("emptyHint").classList.add("hidden");
      draw();
    };
    image.src = `/api/jobs/${data.id}/preview?seq=${data.seq}`;
  }
  const scanning = data.status === "running" || data.status === "stopping" || data.status === "queued";
  $("scanBtn").disabled = scanning || !canStartScan();
  $("stopBtn").disabled = data.status !== "running";
  setStatus(data.message || t("ready"));
  renderResults();
}

function listen(jobId) {
  if (state.source) state.source.close();
  state.source = new EventSource(`/api/jobs/${jobId}/events`);
  const onUpdate = (event) => {
    const data = JSON.parse(event.data);
    applyJob(data);
    if (data.status === "done" || data.status === "error") {
      state.source.close();
      state.source = null;
      if (data.status === "error") setStatus(data.error || t("scanning"));
    }
  };
  state.source.addEventListener("update", onUpdate);
  state.source.addEventListener("done", onUpdate);
  state.source.addEventListener("error", onUpdate);
}

function isLocalCameraRef(value) {
  const text = String(value || "").trim().toLowerCase();
  return /^(?:local|webcam|pc|this-pc|thispc)(?:[:\-]\d+)?$/.test(text) || text.startsWith("local://") || text.startsWith("webcam://");
}

function listedCameraLabel(camera) {
  if (!camera) return "";
  if (camera.custom_label && camera.label) return camera.label;
  if (camera.kind === "local" || isLocalCameraRef(camera.host)) {
    const n = Number(camera.device_index || 0);
    return n ? t("local_camera_n", { n: n + 1 }) : t("local_camera");
  }
  return camera.label || camera.host || "";
}

function isStreamUrl(value) {
  try {
    const parsed = new URL(String(value || "").trim());
    return ["rtsp:", "rtsps:", "http:", "https:"].includes(parsed.protocol) && Boolean(parsed.hostname);
  } catch {
    return false;
  }
}

function isCameraHost(value) {
  const text = String(value || "").trim();
  if (!text || text.includes("://") || text.includes("/") || text.includes("@") || text.includes(" ")) return false;
  return /^(?:\d{1,3}\.){3}\d{1,3}$/.test(text) || /^[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?$/.test(text);
}

function renderCameraSelect() {
  const select = $("ipCameraSelect");
  if (!select) return;
  const current = select.value;
  select.innerHTML = "";
  (state.ipCameras || []).forEach((camera) => {
    const option = document.createElement("option");
    option.value = camera.host;
    option.textContent = listedCameraLabel(camera);
    select.appendChild(option);
  });
  if (!select.options.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = t("ip_camera_add_placeholder");
    select.appendChild(option);
  }
  if (current && [...select.options].some((item) => item.value === current)) select.value = current;
  const selected = (state.ipCameras || []).find((item) => item.host === select.value);
  if ($("ipCameraRemoveBtn")) $("ipCameraRemoveBtn").disabled = !selected || Boolean(selected.builtin);
  if ($("ipCameraSaveBtn")) $("ipCameraSaveBtn").disabled = !selected;
  if (selected && $("ipCameraEditName")) {
    $("ipCameraEditName").value = selected.custom_label ? selected.label : "";
    $("ipCameraEditUrl").value = selected.stream_url || "";
  }
}

function selectedCameraHosts() {
  const allowed = new Set((state.ipCameras || [])
    .filter((camera) => camera.can_scan !== false)
    .map((camera) => camera.host));
  return [...document.querySelectorAll("#cameraSelectorList input[data-camera-toggle]:checked")]
    .map((input) => input.dataset.host)
    .filter((host) => host && allowed.has(host));
}

function cameraTransition(host) {
  return state.cameraTransitions[host] || "";
}

function setCameraTransition(host, phase) {
  if (!host) return;
  if (phase) state.cameraTransitions[host] = phase;
  else delete state.cameraTransitions[host];
  renderCameraSelector();
  renderCameraGrid();
}

function cameraLiveLabel(live, host = "") {
  const transition = cameraTransition(host);
  if (transition === "opening") return t("camera_starting");
  if (transition === "stopping") return t("camera_stopping");
  if (live?.error) return live.error;
  if (live?.starting) return t("camera_starting");
  if (live?.scanning) return t("camera_scanning");
  if (live?.live) return t("camera_live");
  return t("camera_idle");
}

function renderCameraSelector() {
  const list = $("cameraSelectorList");
  if (!list) return;
  const cameras = state.ipCameras || [];
  const active = new Set(liveCameras().filter((camera) => camera.live || camera.starting).map((camera) => camera.host));
  const selectedCount = cameras.filter((camera) => active.has(camera.host)).length;

  if ($("cameraSelectorMeta")) {
    $("cameraSelectorMeta").textContent = t("camera_list_selected", {
      selected: selectedCount,
      total: cameras.length,
    });
  }
  list.innerHTML = cameras.length
    ? cameras.map((camera, index) => {
      const live = liveCamera(camera.host);
      const label = listedCameraLabel(camera) || `Camera ${String(index + 1).padStart(2, "0")}`;
      const transition = cameraTransition(camera.host);
      const checked = Boolean(live?.live || live?.starting || transition === "opening");
      const busy = Boolean(transition);
      const accessLabel = camera.can_scan === false ? t("camera_view_only") : t("camera_scan_allowed");
      return `
        <label class="camera-selector-item${checked ? " is-active" : ""}${busy ? " is-busy" : ""}" ${busy ? 'aria-busy="true"' : ""}>
          <input type="checkbox" data-camera-toggle data-host="${escapeHtml(camera.host)}" ${checked ? "checked" : ""} ${busy ? "disabled" : ""} />
          <span class="camera-selector-check" aria-hidden="true"></span>
          <span class="camera-selector-copy">
            <strong>${escapeHtml(label)}</strong>
            <small>${escapeHtml(camera.host)}</small>
          </span>
          <span class="camera-selector-status">
            <span class="camera-access-badge">${escapeHtml(accessLabel)}</span>
            ${escapeHtml(cameraLiveLabel(live, camera.host))}
          </span>
        </label>`;
    }).join("")
    : `<p class="camera-selector-empty">${escapeHtml(t("ip_camera_add_placeholder"))}</p>`;
}

function cameraRef() {
  const typed = $("ipCameraUrl")?.value.trim() || "";
  if (isStreamUrl(typed) || isCameraHost(typed) || isLocalCameraRef(typed)) return typed;
  return $("ipCameraSelect")?.value || "";
}

function canStartScan() {
  if (state.mode === "camera") return can("scan.camera") && Boolean(cameraRef() || state.ipCameraConfigured);
  if (state.mode === "video") return can("scan.video") && Boolean(state.file);
  return can("scan.image") && Boolean(state.file);
}

function selectedCameraHost() {
  return state.selectedHost || cameraRef();
}

function liveCameras() {
  return Array.isArray(state.worker?.cameras) ? state.worker.cameras : [];
}

function liveCamera(host) {
  return liveCameras().find((item) => item.host === host);
}

function paintComputeChip(gpu) {
  const chip = $("gpuChip");
  if (!chip) return;
  const name = gpu.gpu && gpu.gpu !== "CPU" ? gpu.gpu : "";
  if (gpu.loading) {
    chip.textContent = t("gpu_loading");
    return;
  }
  if (gpu.fallback) {
    chip.textContent = t("compute_fallback");
    return;
  }
  if (gpu.hybrid || gpu.mode === "hybrid") {
    chip.textContent = name ? t("gpu_hybrid", { name }) : t("compute_hybrid");
    return;
  }
  if (name) {
    chip.textContent = t("gpu_ready", { name });
    return;
  }
  chip.textContent = t("gpu_cpu");
}

async function setComputeMode(mode) {
  const select = $("computeMode");
  if (select) select.disabled = true;
  setStatus(t("compute_switching"));
  try {
    const response = await api("/api/worker/compute", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    const text = await response.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { detail: text };
    }
    if (!response.ok) {
      setStatus(typeof data.detail === "string" ? data.detail : t("cam_open_error"));
      return;
    }
    applyWorker(data);
    const gpu = data.gpu || {};
    const modeLabel = t(`compute_${gpu.requested || mode}`);
    setStatus(gpu.fallback ? t("compute_fallback") : t("compute_switched", { mode: modeLabel }));
  } finally {
    if (select) select.disabled = false;
  }
}

function applyWorker(data) {
  state.worker = data || { cameras: [], gpu: {}, plates: [] };
  const cameras = liveCameras();
  state.liveOpen = cameras.some((item) => item.live || item.starting);
  state.liveRecording = cameras.some((item) => item.recording);
  if (Array.isArray(data.plates) && data.plates.length) {
    const previousKey = livePlateKey(state.plates[state.selected]);
    const previousCount = state.plates.length;
    mergeLivePlates(data.plates);
    if (state.plates.length > previousCount) {
      state.selected = state.plates.length - 1;
    } else if (previousKey) {
      const idx = state.plates.findIndex((plate) => livePlateKey(plate) === previousKey);
      state.selected = idx >= 0 ? idx : Math.max(0, state.plates.length - 1);
    } else if (state.plates.length) {
      state.selected = state.plates.length - 1;
    }
  }
  const dragging = state.drag?.host;
  cameras.forEach((camera) => {
    if (!camera.host || !camera.roi || state.cameraRois[camera.host]) return;
    if (dragging && dragging === camera.host) return;
    state.cameraRois[camera.host] = copyRoi(camera.roi);
  });
  const gpu = data.gpu || {};
  paintComputeChip(gpu);
  if ($("computeMode") && gpu.requested && $("computeMode").value !== gpu.requested && !$("computeMode").disabled) {
    $("computeMode").value = gpu.requested;
  }
  const liveCount = cameras.filter((item) => item.live || item.starting).length;
  $("cameraLiveStatus").textContent = liveCount ? `${t("worker_head")} · ${liveCount}` : "";
  const selectedCamera = liveCamera(selectedCameraHost());
  $("camStart").textContent = selectedCamera?.live || selectedCamera?.starting ? t("cam_close") : t("cam_start");
  const selected = liveCamera(selectedCameraHost());
  $("camSnap").disabled = !selected?.live;
  $("camRecord").disabled = !selected?.live || Boolean(selected?.recording);
  $("camStopRec").disabled = !selected?.recording;
  $("scanBtn").disabled = !canStartScan();
  $("stopBtn").disabled = !state.liveOpen;
  $("plateCount").textContent = t("plates", { n: state.plates.length });
  renderCameraSelector();
  renderCameraGrid();
  renderResults();
}

function cameraWallLayout(count, width, height) {
  const gap = 8;
  const minW = width < 520 ? 150 : 180;
  const minH = height < 280 ? 110 : 130;
  const cameras = Math.max(1, count);
  const fits = (size) => {
    let best = null;
    for (let cols = 1; cols <= size; cols += 1) {
      const rows = Math.ceil(size / cols);
      const cellW = (width - gap * (cols + 1)) / cols;
      const cellH = (height - gap * (rows + 1)) / rows;
      if (cellW < minW || cellH < minH) continue;
      const area = cellW * cellH;
      if (!best || area > best.area) best = { cols, rows, area };
    }
    return best;
  };
  let pageSize = cameras;
  let layout = fits(pageSize);
  while (!layout && pageSize > 1) {
    pageSize -= 1;
    layout = fits(pageSize);
  }
  return {
    pageSize: layout ? pageSize : 1,
    cols: layout?.cols || 1,
    rows: layout?.rows || 1,
  };
}

function bindCameraStream(tile, host, live) {
  const img = tile.querySelector("img");
  if (!img) return;
  const offpage = tile.classList.contains("is-offpage");
  if (live?.live && !offpage) {
    if (img.dataset.live !== "1") {
      img.dataset.live = "1";
      img.decoding = "async";
      img.src = `/api/cameras/live.mjpeg?host=${encodeURIComponent(host)}&n=${Date.now()}`;
    }
    return;
  }
  if (img.dataset.live === "1") {
    img.dataset.live = "0";
    img.removeAttribute("src");
  }
}

function bindVisibleCameraStreams() {
  const grid = $("cameraGrid");
  if (!grid) return;
  [...grid.querySelectorAll("[data-host]")].forEach((tile) => {
    bindCameraStream(tile, tile.dataset.host, liveCamera(tile.dataset.host));
  });
}

function activeCameraList() {
  const activeHosts = new Set(liveCameras().filter((camera) => camera.live || camera.starting).map((camera) => camera.host));
  return (state.ipCameras || []).filter((camera) => activeHosts.has(camera.host));
}

function layoutCameraWall() {
  const grid = $("cameraGrid");
  const pager = $("cameraWallPager");
  const meta = $("cameraWallMeta");
  if (!grid || state.mode !== "camera") return;
  const listed = activeCameraList();
  const total = listed.length;
  const width = grid.clientWidth || 640;
  const height = grid.clientHeight || 320;
  const wall = cameraWallLayout(Math.max(1, total), width, height);
  const pages = Math.max(1, Math.ceil(Math.max(total, 1) / wall.pageSize));
  if (state.cameraPage >= pages) state.cameraPage = 0;
  const start = state.cameraPage * wall.pageSize;
  const visible = listed.slice(start, start + wall.pageSize);
  const visibleCount = Math.max(1, visible.length);
  const pageLayout = cameraWallLayout(visibleCount, width, height);
  grid.style.setProperty("--wall-cols", String(pageLayout.cols));
  grid.style.setProperty("--wall-rows", String(Math.max(pageLayout.rows, Math.ceil(visibleCount / pageLayout.cols))));
  grid.dataset.pageSize = String(wall.pageSize);
  const known = new Set(visible.map((camera) => camera.host));
  [...grid.querySelectorAll("[data-host]")].forEach((tile) => {
    tile.classList.toggle("is-offpage", !known.has(tile.dataset.host));
  });
  if (meta) meta.textContent = total ? t("wall_cameras", { n: total }) : "";
  if (pager) {
    const many = pages > 1;
    pager.hidden = !many;
    pager.classList.toggle("hidden", !many);
    if ($("cameraWallPageLabel")) {
      $("cameraWallPageLabel").textContent = t("wall_page", { page: state.cameraPage + 1, pages });
    }
    if ($("cameraWallPrev")) $("cameraWallPrev").disabled = !many;
    if ($("cameraWallNext")) $("cameraWallNext").disabled = !many;
  }
  visible.forEach((camera) => drawCameraRoi(camera.host));
  bindVisibleCameraStreams();
}

function shiftCameraWall(delta) {
  const listed = activeCameraList();
  const pageSize = Number($("cameraGrid")?.dataset.pageSize || listed.length) || 1;
  const pages = Math.max(1, Math.ceil(Math.max(listed.length, 1) / pageSize));
  state.cameraPage = (state.cameraPage + delta + pages) % pages;
  layoutCameraWall();
}

function renderCameraGrid() {
  const grid = $("cameraGrid");
  if (!grid || state.mode !== "camera") return;
  ensureCameraRois();
  const listed = activeCameraList();
  const known = new Set();
  const selected = selectedCameraHost();
  listed.forEach((camera, order) => {
    const host = camera.host;
    known.add(host);
    let tile = grid.querySelector(`[data-host="${CSS.escape(host)}"]`);
    if (!tile) {
      tile = document.createElement("article");
      tile.className = "camera-tile";
      tile.dataset.host = host;
      tile.innerHTML = `
        <div class="camera-tile-stage">
          <img alt="" decoding="async" />
          <canvas class="camera-roi"></canvas>
          <div class="camera-tile-chrome">
            <div class="camera-tile-head">
              <span class="camera-tile-index"></span>
              <span class="camera-tile-host"></span>
            </div>
            <div>
              <div class="camera-tile-meta">
                <span data-role="status"></span>
                <strong data-role="plate"></strong>
              </div>
              <div class="camera-tile-actions">
                <button type="button" class="btn ghost" data-action="snap"></button>
              </div>
            </div>
          </div>
        </div>`;
      bindCameraRoi(tile.querySelector("canvas.camera-roi"), host);
      tile.querySelector("img").addEventListener("load", () => drawCameraRoi(host));
      tile.addEventListener("click", (event) => {
        const button = event.target.closest("[data-action]");
        selectCameraHost(host);
        if (!button) return;
        if (button.dataset.action === "snap") {
          event.preventDefault();
          snapshot(host).catch((error) => setStatus(String(error)));
        }
      });
      grid.appendChild(tile);
    }
    const live = liveCamera(host);
    const label = listedCameraLabel(camera) || live?.label || `Camera ${String(camera.index || order + 1).padStart(2, "0")}`;
    tile.querySelector(".camera-tile-index").textContent = label;
    tile.querySelector(".camera-tile-host").textContent = host;
    tile.querySelector("[data-role=status]").textContent = live?.error
      ? live.error
      : live?.starting
        ? t("camera_starting")
      : live?.scanning
        ? Number(live.sampled_frames) > 0
          ? live.last_detect_count
            ? t("camera_scanning_found", { n: live.last_detect_count })
            : t("camera_scanning")
          : t("camera_scanning_load")
        : live?.live
          ? t("camera_live")
          : t("camera_idle");
    tile.querySelector("[data-role=plate]").textContent = live?.last_plate || "";
    tile.querySelector("[data-action=snap]").textContent = t("cam_snap");
    tile.querySelector("[data-action=snap]").disabled = !live?.live;
    tile.classList.toggle("is-live", Boolean(live?.live));
    tile.classList.toggle("is-starting", Boolean(live?.starting));
    tile.classList.toggle("is-error", Boolean(live?.error));
    tile.classList.toggle("is-selected", host === selected);
    const img = tile.querySelector("img");
    if (img) img.decoding = "async";
  });
  [...grid.querySelectorAll("[data-host]")].forEach((tile) => {
    if (!known.has(tile.dataset.host)) tile.remove();
  });
  layoutCameraWall();
}

function workerSocketUrl() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/ws`;
}

function startWorkerEvents() {
  state.workerEventsWanted = true;
  if (state.workerEventsSuspended || state.workerSocket || !window.WebSocket) return;
  if (state.workerSocketTimer) {
    window.clearTimeout(state.workerSocketTimer);
    state.workerSocketTimer = 0;
  }
  const socket = new WebSocket(workerSocketUrl());
  state.workerSocket = socket;
  socket.onopen = () => {
    state.workerSocketRetry = 0;
  };
  socket.onmessage = (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (payload.type === "ROI_UPDATED") {
        const host = String(payload.data?.host || "");
        const roi = payload.data?.roi;
        if (!roi) return;
        if (host) {
          state.cameraRois[host] = copyRoi(roi);
          if (host === selectedCameraHost()) state.roi = copyRoi(roi);
          drawCameraRoi(host);
        } else {
          state.roi = copyRoi(roi);
        }
        syncRoiToolbar();
        return;
      }
      if (payload.type === "WORKER_UPDATED") {
        const snapshot = payload.data?.snapshot;
        if (snapshot) applyWorker(snapshot);
      }
    } catch {
      // Ignore malformed or non-worker events and keep the stream alive.
    }
  };
  socket.onerror = () => socket.close();
  socket.onclose = (event) => {
    if (state.workerSocket !== socket) return;
    state.workerSocket = null;
    if (!state.workerEventsWanted || state.workerEventsSuspended) return;
    if (event.code === 4001 || event.code === 1008) {
      fetch("/api/auth/me", { credentials: "same-origin" }).then((response) => {
        if (!response.ok) window.location.replace("/login");
      }).catch(() => {});
      return;
    }
    const delay = Math.min(10000, 1000 * (2 ** Math.min(state.workerSocketRetry, 3)));
    state.workerSocketRetry += 1;
    state.workerSocketTimer = window.setTimeout(() => {
      state.workerSocketTimer = 0;
      startWorkerEvents();
    }, delay);
  };
}

function suspendWorkerEvents() {
  state.workerEventsSuspended = true;
  if (state.workerSocketTimer) window.clearTimeout(state.workerSocketTimer);
  state.workerSocketTimer = 0;
  if (state.workerSocket) {
    const socket = state.workerSocket;
    state.workerSocket = null;
    socket.close();
  }
}

function resumeWorkerEvents() {
  state.workerEventsSuspended = false;
  if (!state.workerEventsWanted) return;
  startWorkerEvents();
  refreshWorker().catch(() => {
    // The socket will continue reconnecting if the page was restored offline.
  });
}

function stopWorkerEvents(stopStreams = false) {
  state.workerEventsWanted = false;
  if (state.workerSocketTimer) window.clearTimeout(state.workerSocketTimer);
  state.workerSocketTimer = 0;
  state.workerSocketRetry = 0;
  if (state.workerSocket) {
    const socket = state.workerSocket;
    state.workerSocket = null;
    socket.close();
  }
  if (stopStreams) {
    Object.values(state.workerBlobs).forEach((url) => URL.revokeObjectURL(url));
    state.workerBlobs = {};
  }
}

async function refreshWorker() {
  const response = await api("/api/worker");
  if (!response.ok) return;
  applyWorker(await response.json());
}

async function startCameraHosts(hosts) {
  if (hosts.length > 1) {
    const rois = {};
    hosts.forEach((host) => { rois[host] = roiFor(host); });
    const response = await api("/api/worker/start-all", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hosts, rois }),
    });
    const text = await response.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { detail: text || `HTTP ${response.status}` };
    }
    await refreshWorker();
    const errors = new Map((data.errors || []).map((item) => [item.host, item.error || data.detail]));
    return hosts.map((host) => ({
      host,
      ok: response.ok && !errors.has(host),
      data: errors.has(host) ? { detail: errors.get(host) } : data,
    }));
  }
  const results = await Promise.all(hosts.map(async (host) => {
    const response = await api("/api/worker/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ host, roi: roiFor(host) }),
    });
    const text = await response.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { detail: text || `HTTP ${response.status}` };
    }
    return { host, ok: response.ok, data };
  }));
  await refreshWorker();
  return results;
}

async function startIpCameraScan() {
  if (!can("scan.camera")) {
    setStatus(t("no_permission"));
    return;
  }
  if (!(state.ipCameras || []).length) {
    setStatus(t("ip_camera_add_placeholder"));
    return;
  }
  const selected = selectedCameraHosts();
  if (!selected.length) {
    setStatus(t("camera_enable"));
    return;
  }
  $("scanBtn").disabled = true;
  $("stopBtn").disabled = false;
  state.plates = [];
  state.selected = -1;
  renderResults();
  setStatus(t("scanning"));
  const results = await startCameraHosts(selected);
  const failed = results.filter((item) => !item.ok).map((item) => item.host);
  if (failed.length === selected.length) {
    const detail = results[0]?.data?.detail;
    setStatus(typeof detail === "string" ? detail : t("cam_open_error"));
    $("scanBtn").disabled = false;
    $("stopBtn").disabled = true;
    return;
  }
  setStatus(failed.length ? failed.join(", ") : t("scanning"));
  startWorkerEvents();
}

async function stopScan() {
  if (state.mode === "camera") {
    await api("/api/worker/stop-all", { method: "POST" });
    await refreshWorker();
    setStatus(t("ready"));
    return;
  }
  if (!state.jobId) return;
  await api(`/api/jobs/${state.jobId}/stop`, { method: "POST" });
}

async function toggleCamera(host) {
  const camera = (state.ipCameras || []).find((item) => item.host === host);
  if (!camera || camera.can_view === false) {
    setStatus(t("no_permission"));
    return;
  }
  if (cameraTransition(host)) return;
  const live = liveCamera(host);
  if (live?.live || live?.starting) {
    await stopCamera(host);
    return;
  }
  await openCamera(host);
}

async function setCameraSelection(host, enabled) {
  selectCameraHost(host);
  if (cameraTransition(host)) return;
  if (enabled) {
    await openCamera(host);
    return;
  }
  if (liveCamera(host)?.live || liveCamera(host)?.starting) {
    await stopCamera(host);
    return;
  }
  renderCameraSelector();
  renderCameraGrid();
}

async function setAllCameraSelections(enabled) {
  const cameras = (state.ipCameras || []).filter((camera) => camera.host);
  const scanHosts = cameras.filter((camera) => camera.can_scan !== false).map((camera) => camera.host);
  const viewHosts = cameras.filter((camera) => camera.can_scan === false).map((camera) => camera.host);
  if (!cameras.length) return;
  if (enabled) {
    scanHosts.forEach((host) => setCameraTransition(host, "opening"));
    try {
      const results = scanHosts.length ? await startCameraHosts(scanHosts) : [];
      const failed = results.filter((item) => !item.ok);
      if (failed.length) {
        const detail = failed[0]?.data?.detail;
        setStatus(typeof detail === "string" ? detail : t("cam_open_error"));
      }
      await Promise.all(viewHosts.map((host) => openCamera(host)));
      startWorkerEvents();
    } finally {
      scanHosts.forEach((host) => setCameraTransition(host, ""));
    }
  } else {
    const active = cameras.map((camera) => camera.host).filter((host) => {
      const live = liveCamera(host);
      return live?.live || live?.starting;
    });
    active.forEach((host) => setCameraTransition(host, "stopping"));
    try {
      const response = await api("/api/worker/stop-all", { method: "POST" });
      if (!response.ok) throw new Error((await response.text()) || `HTTP ${response.status}`);
      await refreshWorker();
    } finally {
      active.forEach((host) => setCameraTransition(host, ""));
    }
  }
  setStatus(enabled ? t("scanning") : t("ready"));
}

async function stopCamera(host) {
  if (!host || cameraTransition(host)) return;
  setCameraTransition(host, "stopping");
  try {
    const response = await api("/api/worker/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ host }),
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `HTTP ${response.status}`);
    }
    await refreshWorker();
    setStatus(t("ready"));
  } finally {
    setCameraTransition(host, "");
  }
}

async function openCamera(host) {
  const camera = (state.ipCameras || []).find((item) => item.host === (host || cameraRef()));
  if (!camera || camera.can_view === false) {
    setStatus(t("no_permission"));
    return;
  }
  const target = host || cameraRef();
  if (!target) {
    setStatus(t("ip_camera_add_placeholder"));
    return;
  }
  if (cameraTransition(target)) return;
  const current = liveCamera(target);
  if (current?.live && !host) {
    await toggleCamera(target);
    return;
  }
  const scansPlates = camera.can_scan !== false && can("scan.camera");
  setCameraTransition(target, "opening");
  state.liveOpening = true;
  $("camStart").disabled = true;
  setStatus(scansPlates ? t("scanning") : t("cam_live", { host: target }));
  try {
    const response = await api(scansPlates ? "/api/worker/start" : "/api/cameras/live", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ host: target, roi: roiFor(target) }),
    });
    const text = await response.text();
    let data = {};
    try {
      data = text ? JSON.parse(text) : {};
    } catch {
      data = { detail: text || `HTTP ${response.status}` };
    }
    if (!response.ok) {
      const detail = data.detail;
      setStatus(typeof detail === "string" ? detail : t("cam_open_error"));
      await refreshWorker().catch(() => {});
      return;
    }
    if (data.error) {
      setStatus(data.error);
      await refreshWorker().catch(() => {});
      return;
    }
    state.selectedHost = data.host || target;
    await refreshWorker();
    setStatus(data.starting ? t("camera_starting") : scansPlates ? t("scanning") : t("cam_live", { host: data.host || target }));
    startWorkerEvents();
  } finally {
    setCameraTransition(target, "");
    state.liveOpening = false;
    $("camStart").disabled = false;
  }
}

async function stopLivePreview() {
  const host = selectedCameraHost();
  if (host) await stopCamera(host);
}

async function snapshot(host) {
  const target = host || selectedCameraHost();
  if (!liveCamera(target)?.live) {
    setStatus(t("cam_open_error"));
    return;
  }
  const response = await api(`/api/cameras/live/snapshot.jpg?host=${encodeURIComponent(target)}`);
  if (!response.ok) {
    setStatus(t("cam_open_error"));
    return;
  }
  const blob = await response.blob();
  acceptFile(new File([blob], `${target}.jpg`, { type: "image/jpeg" }));
  setMode("image");
  setStatus(t("cam_snap_ok"));
}

async function startRecord() {
  const host = selectedCameraHost();
  if (!liveCamera(host)?.live) return;
  const response = await api(`/api/cameras/live/record?host=${encodeURIComponent(host)}`, { method: "POST" });
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { detail: text };
  }
  if (!response.ok) {
    setStatus(typeof data.detail === "string" ? data.detail : t("cam_open_error"));
    return;
  }
  state.liveRecording = true;
  $("camRecord").disabled = true;
  $("camStopRec").disabled = false;
  $("cameraLiveStatus").textContent = t("cam_recording", { host: data.host || host });
  setStatus(t("cam_recording", { host: data.host || host }));
}

async function stopRecord() {
  const host = selectedCameraHost();
  const response = await api(`/api/cameras/live/record/stop?host=${encodeURIComponent(host)}`, { method: "POST" });
  $("camRecord").disabled = !liveCamera(host)?.live;
  $("camStopRec").disabled = true;
  state.liveRecording = false;
  if (!response.ok) {
    const text = await response.text();
    setStatus(text || t("cam_open_error"));
    return;
  }
  const blob = await response.blob();
  const name = (response.headers.get("content-disposition") || "").match(/filename="?([^";]+)/i)?.[1] || "camera.mp4";
  acceptFile(new File([blob], name, { type: blob.type || "video/mp4" }));
  setMode("video");
  setStatus(t("cam_record_ok"));
}

async function loadHealth() {
  try {
    const data = await (await fetch("/api/health")).json();
    $("healthChip").textContent = data.database === "ok" ? t("db_ok") : data.database === "missing" ? t("db_off") : data.database;
    if ($("gpuChip")) {
      paintComputeChip({
        loading: data.gpu_loading,
        gpu: data.gpu,
        mode: data.compute,
        requested: data.compute,
        hybrid: data.compute_hybrid,
        fallback: data.compute === "cpu" && data.gpu === "CPU",
        yolo_device: data.compute_yolo,
        ocr_device: data.compute_ocr,
      });
    }
    if ($("computeMode") && data.compute) $("computeMode").value = data.compute;
    state.ipCameras = Array.isArray(data.ip_cameras) ? data.ip_cameras : [];
    state.cameraAccessAvailable = state.ipCameras.length > 0;
    state.cameraStorage = data.camera_storage || (data.database === "ok" ? "database" : "file");
    if ($("cameraStorageStatus")) {
      $("cameraStorageStatus").textContent = state.cameraStorage === "database"
        ? t("camera_storage_database")
        : t("camera_storage_file");
    }
    ensureCameraRois();
    state.ipCameraConfigured = state.ipCameras.length > 0 || Boolean(data.ip_camera);
    renderCameraSelect();
    applyPermissions();
    if (state.mode === "camera") renderCameraGrid();
    if ($("ipCameraUrl")) $("ipCameraUrl").placeholder = t("ip_camera_add_placeholder");
    if (state.mode === "camera") $("scanBtn").disabled = !canStartScan();
  } catch {
    $("healthChip").textContent = t("db_off");
  }
}

async function startScan() {
  if (state.mode === "camera") {
    await startIpCameraScan();
    return;
  }
  if (!state.file) return;
  const mediaType = state.file.type.startsWith("video/") || state.mode === "video" ? "video" : "image";
  if (!can(mediaType === "video" ? "scan.video" : "scan.image")) {
    setStatus(t("no_permission"));
    return;
  }
  const file = state.file;
  const filename = file.name || (file.type.startsWith("video/") ? "upload.webm" : "upload.jpg");
  const body = new FormData();
  body.append("file", file, filename);
  body.append("media_type", file.type.startsWith("video/") || state.mode === "video" ? "video" : "image");
  body.append("roi", JSON.stringify(state.roi));
  $("scanBtn").disabled = true;
  $("stopBtn").disabled = false;
  setStatus(t("scanning"));
  const response = await api("/api/jobs", { method: "POST", body });
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { detail: text || `HTTP ${response.status}` };
  }
  if (!response.ok) {
    const detail = data.detail;
    setStatus(typeof detail === "string" ? detail : JSON.stringify(detail || data));
    $("scanBtn").disabled = false;
    $("stopBtn").disabled = true;
    return;
  }
  state.jobId = data.id;
  applyJob(data);
  listen(data.id);
}

function saveJson() {
  if (!state.result || !can("results.save")) {
    setStatus(t("no_permission"));
    return;
  }
  const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "scan_result.json";
  link.click();
  URL.revokeObjectURL(url);
}

async function addLocalCamera() {
  $("ipCameraUrl").value = "local";
  await addIpCamera();
}

async function addIpCamera() {
  const host = $("ipCameraUrl").value.trim();
  const label = $("ipCameraName")?.value.trim() || "";
  const username = $("ipCameraUser")?.value.trim() || "";
  const password = $("ipCameraPassword")?.value || "";
  const path = $("ipCameraPath")?.value.trim() || "";
  if (!host) {
    setStatus(t("ip_camera_add_placeholder"));
    return;
  }
  const response = await api("/api/cameras", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ host, username, password, path, label }),
  });
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { detail: text || `HTTP ${response.status}` };
  }
  if (!response.ok) {
    const detail = data.detail;
    setStatus(typeof detail === "string" ? detail : t("ip_camera_add_error"));
    return;
  }
  state.ipCameras = Array.isArray(data.cameras) ? data.cameras : state.ipCameras;
  state.ipCameraConfigured = state.ipCameras.length > 0;
  if (data.host && $("ipCameraSelect")) $("ipCameraSelect").value = data.host;
  renderCameraSelect();
  $("ipCameraUrl").value = "";
  if ($("ipCameraName")) $("ipCameraName").value = "";
  if ($("ipCameraUser")) $("ipCameraUser").value = "";
  if ($("ipCameraPassword")) $("ipCameraPassword").value = "";
  if ($("ipCameraPath")) $("ipCameraPath").value = "";
  setStatus(t("ip_camera_added", { host: data.host || host }));
  if (state.mode === "camera") {
    $("scanBtn").disabled = !canStartScan();
    renderCameraSelector();
    renderCameraGrid();
  }
}

async function saveIpCamera() {
  const host = $("ipCameraSelect")?.value;
  if (!host) return;
  const response = await api(`/api/cameras?host=${encodeURIComponent(host)}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      label: $("ipCameraEditName").value.trim(),
      stream_url: $("ipCameraEditUrl").value.trim(),
    }),
  });
  const text = await response.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch { data = { detail: text || `HTTP ${response.status}` }; }
  if (!response.ok) {
    const detail = data.detail;
    setStatus(typeof detail === "string" ? detail : t("ip_camera_add_error"));
    return;
  }
  state.ipCameras = Array.isArray(data.cameras) ? data.cameras : state.ipCameras;
  renderCameraSelect();
  if (state.mode === "camera") renderCameraGrid();
  setStatus(t("ip_camera_saved", { host }));
}

async function removeIpCamera() {
  const host = $("ipCameraSelect")?.value;
  if (!host) return;
  const response = await api(`/api/cameras?host=${encodeURIComponent(host)}`, { method: "DELETE" });
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { detail: text || `HTTP ${response.status}` };
  }
  if (!response.ok) {
    const detail = data.detail;
    setStatus(typeof detail === "string" ? detail : t("ip_camera_add_error"));
    return;
  }
  if (liveCamera(host)?.live) {
    await api("/api/worker/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ host }),
    });
  }
  state.ipCameras = Array.isArray(data.cameras) ? data.cameras : state.ipCameras;
  state.ipCameraConfigured = state.ipCameras.length > 0;
  renderCameraSelect();
  setStatus(t("ip_camera_removed", { host }));
  if (state.mode === "camera") {
    $("scanBtn").disabled = !canStartScan();
    renderCameraGrid();
  }
}

async function loadMe() {
  const response = await fetch("/api/auth/me");
  if (response.status === 401) {
    window.location.href = "/login";
    return false;
  }
  state.user = await response.json();
  try {
    const catalogResponse = await fetch("/api/auth/catalog");
    if (catalogResponse.ok) {
      const catalog = await catalogResponse.json();
      state.roleCatalog = Array.isArray(catalog.roles) ? catalog.roles : [];
      state.statusCatalog = Array.isArray(catalog.statuses) ? catalog.statuses : [];
      renderRoleCatalog();
    }
  } catch (_) {
    // The built-in role options remain available when the catalog endpoint is unavailable.
  }
  applyPermissions();
  startWorkerEvents();
  return true;
}

function renderUsers() {
  $("usersHeader").innerHTML = [t("col_user"), t("col_name"), t("col_role"), t("col_status"), t("col_actions")]
    .map((label) => `<th>${label}</th>`)
    .join("");
  const selfId = Number(state.user?.id);
  $("usersBody").innerHTML = state.users
    .map((user) => {
      const toggle = user.active ? t("action_disable") : t("action_enable");
      const isSelf = Number(user.id) === selfId;
      const deleteBtn = isSelf
        ? ""
        : `<button type="button" class="btn ghost" data-user-delete="${user.id}" data-user-name="${escapeHtml(user.display_name || user.username)}">${t("user_delete")}</button>`;
      return `<tr>
        <td>${escapeHtml(user.username)}</td>
        <td>${escapeHtml(user.display_name)}</td>
        <td>${roleLabel(user.role)}</td>
        <td>${user.active ? t("status_active") : t("status_disabled")}</td>
        <td class="users-actions">
          <button type="button" class="btn ghost" data-user-edit="${user.id}">${t("user_edit")}</button>
          <button type="button" class="btn ghost" data-user-toggle="${user.id}" data-active="${user.active ? "1" : "0"}">${toggle}</button>
          ${deleteBtn}
        </td>
      </tr>`;
    })
    .join("");
}

async function loadUsers() {
  const response = await api("/api/users");
  const data = await response.json();
  if (!response.ok) {
    $("usersError").textContent = data.detail || t("no_permission");
    $("usersError").hidden = false;
    return;
  }
  state.users = data.users || [];
  renderUsers();
}

function openCameras() {
  if (!can("scan.camera")) {
    setStatus(t("no_permission"));
    return;
  }
  closeUsers();
  closeHistory();
  closeReport();
  $("camerasPanel").hidden = false;
  $("camerasPanel").classList.remove("hidden");
  startWorkerEvents();
  $("camerasBtn").setAttribute("aria-pressed", "true");
  renderCameraSelect();
  refreshWorker().catch(() => {});
}

function closeCameras() {
  if (!$("camerasPanel")) return;
  $("camerasPanel").hidden = true;
  $("camerasPanel").classList.add("hidden");
  $("camerasBtn").setAttribute("aria-pressed", "false");
  if (state.mode !== "camera" && !state.liveOpen && !state.liveOpening) stopWorkerEvents(false);
}

function openUsers() {
  if (!can("users.manage")) {
    setStatus(t("no_permission"));
    return;
  }
  closeCameras();
  closeHistory();
  closeReport();
  $("usersPanel").hidden = false;
  $("usersPanel").classList.remove("hidden");
  $("usersError").hidden = true;
  $("usersBtn").setAttribute("aria-pressed", "true");
  loadUsers().catch((error) => setStatus(String(error)));
}

function closeUsers() {
  $("usersPanel").hidden = true;
  $("usersPanel").classList.add("hidden");
  $("usersBtn").setAttribute("aria-pressed", "false");
  cancelEditUser();
}

function operatorLabel(group) {
  const name = group.operator_name || group.display_name || "";
  const username = group.operator_username || group.username || "";
  if (name && username && name !== username) return `${name} (${username})`;
  return name || username || t("operator_unknown");
}

function passagePlate(scan) {
  if (scan.plate_prefix || scan.plate_number || scan.ocr_text) return scan;
  return (scan.plates && scan.plates[0]) || scan;
}

function plateLine(scan) {
  const plate = passagePlate(scan);
  const prefix = plate.plate_prefix || "";
  const number = plate.plate_number || "";
  if (prefix && number) return `${prefix}-${number}`;
  return number || prefix || plate.ocr_text || t("plate_empty");
}

function renderHistory(data) {
  const errorBox = $("historyError");
  const emptyBox = $("historyEmpty");
  const body = $("historyBody");
  errorBox.hidden = true;
  emptyBox.hidden = true;
  body.innerHTML = "";
  if (data.database && data.database !== "ok") {
    emptyBox.textContent = data.database === "missing" ? t("history_db_off") : data.database;
    emptyBox.hidden = false;
    return;
  }
  const select = $("historyOperator");
  const current = select.value;
  select.innerHTML = `<option value="">${t("history_all_operators")}</option>`;
  (data.operators || []).forEach((operator) => {
    const option = document.createElement("option");
    option.value = operator.id == null ? "unassigned" : String(operator.id);
    option.textContent = operatorLabel(operator);
    select.append(option);
  });
  if ([...select.options].some((option) => option.value === current)) select.value = current;
  const searching = Boolean(data.query);
  const groups = data.groups || [];
  if (!groups.length) {
    emptyBox.textContent = searching ? t("history_empty_search") : t("history_empty");
    emptyBox.hidden = false;
    return;
  }
  body.innerHTML = groups
    .map((group) => {
      const rows = group.scans
        .map((scan) => {
          const when = searching && scan.date ? `${scan.date} ${scan.scanned_time || ""}` : scan.scanned_time || "";
          const photo = scan.vehicle_url
            ? `<img class="ledger-thumb" src="${scan.vehicle_url}" alt="" loading="lazy" />`
            : `<span class="ledger-thumb empty">${t("history_no_image")}</span>`;
          const plateId = scan.plate_id == null ? "" : String(scan.plate_id);
          return `<tr data-scan-id="${scan.scan_id || scan.id}" data-plate-id="${plateId}">
            <td>${photo}</td>
            <td>${when}</td>
            <td>${plateLine(scan)}</td>
            <td>${countryLabel(scan.country)}</td>
            <td>${scan.vehicle_type || "-"}</td>
            <td>${scan.province || "-"}</td>
          </tr>`;
        })
        .join("");
      return `<section class="ledger-shift">
        <header>
          <span>${operatorLabel(group)}</span>
          <span>${t("shift_count", { n: group.scan_count })}</span>
        </header>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>${t("col_photo")}</th><th>${t("col_time")}</th><th>${t("col_plates")}</th><th>${t("country")}</th><th>${t("vehicle")}</th><th>${t("province")}</th></tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>
        </div>
      </section>`;
    })
    .join("");
}

async function loadHistory() {
  const day = $("historyDate").value;
  const operator = $("historyOperator").value;
  const query = $("historyQuery").value.trim();
  const params = new URLSearchParams();
  if (query) params.set("q", query);
  else if (day) params.set("day", day);
  if (operator) params.set("operator_id", operator);
  const response = await api(`/api/scans?${params.toString()}`);
  const data = await response.json();
  if (!response.ok) {
    $("historyError").textContent = data.detail || t("no_permission");
    $("historyError").hidden = false;
    return;
  }
  renderHistory(data);
}

function closeScanSlip() {
  $("historySlip").hidden = true;
  $("historySlip").classList.add("hidden");
  $("historyPanel").querySelector(".ledger-card").classList.remove("has-slip");
  $("historyBody").querySelectorAll("tr.selected").forEach((row) => row.classList.remove("selected"));
}

async function openScanSlip(scanId, plateId, row) {
  $("historyBody").querySelectorAll("tr.selected").forEach((item) => item.classList.remove("selected"));
  if (row) row.classList.add("selected");
  const slip = $("historySlip");
  slip.hidden = false;
  slip.classList.remove("hidden");
  $("historyPanel").querySelector(".ledger-card").classList.add("has-slip");
  $("historySlipMeta").textContent = "";
  $("historySlipDetails").innerHTML = "";
  $("historySlipImage").hidden = true;
  $("historySlipCrop").hidden = true;
  $("historySlipNoImage").hidden = true;
  const response = await api(`/api/scans/${scanId}`);
  const scan = await response.json();
  if (!response.ok) {
    $("historyError").textContent = scan.detail || t("no_permission");
    $("historyError").hidden = false;
    return;
  }
  const plates = scan.plates || [];
  const plate = plates.find((item) => String(item.id) === String(plateId)) || plates[0] || {};
  $("historySlipMeta").textContent = `${scan.date || ""} ${scan.scanned_time || ""} · ${operatorLabel(scan)}`;
  const vehicleUrl = plate.vehicle_url || scan.image_url;
  const img = $("historySlipImage");
  const crop = $("historySlipCrop");
  if (vehicleUrl) {
    img.hidden = false;
    img.alt = plateLine(plate);
    img.src = `${vehicleUrl}?t=${Date.now()}`;
    $("historySlipNoImage").hidden = true;
  } else {
    img.removeAttribute("src");
    img.hidden = true;
    $("historySlipNoImage").hidden = false;
  }
  if (plate.crop_url) {
    crop.hidden = false;
    crop.alt = plateLine(plate);
    crop.src = `${plate.crop_url}?t=${Date.now()}`;
  } else {
    crop.removeAttribute("src");
    crop.hidden = true;
  }
  const fields = [
    [t("country"), countryLabel(plate.country)],
    [t("vehicle"), plate.vehicle_type || t("unknown")],
    [t("plate_type"), plate.plate_type || "-"],
    [t("prefix"), plate.plate_prefix || "-"],
    [t("number"), plate.plate_number || "-"],
    [t("province"), plate.province || t("unknown")],
    [t("province_code"), plate.province_code || "-"],
    [t("ocr"), plate.ocr_text || "-"],
    [t("confidence"), plate.confidence ? `${(Number(plate.confidence) * 100).toFixed(1)}%` : "-"],
    [t("media"), scan.media_type || "-"],
    [t("source_file"), scan.source_name || "-"],
  ];
  $("historySlipDetails").innerHTML = fields
    .map(([label, value]) => `<dt>${label}</dt><dd>${value}</dd>`)
    .join("");
}

function openHistory() {
  closeCameras();
  closeUsers();
  closeReport();
  if (!$("historyDate").value) {
    const now = new Date();
    const local = new Date(now.getTime() - now.getTimezoneOffset() * 60000);
    $("historyDate").value = local.toISOString().slice(0, 10);
  }
  $("historyPanel").hidden = false;
  $("historyPanel").classList.remove("hidden");
  $("historyBtn").setAttribute("aria-pressed", "true");
  loadHistory().catch((error) => setStatus(String(error)));
}

function closeHistory() {
  closeScanSlip();
  $("historyPanel").hidden = true;
  $("historyPanel").classList.add("hidden");
  $("historyBtn").setAttribute("aria-pressed", "false");
}

function localIsoDate(value = new Date()) {
  const date = value instanceof Date ? value : new Date(value);
  const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 10);
}

function shiftIsoDate(iso, days) {
  const date = new Date(`${iso}T00:00:00`);
  date.setDate(date.getDate() + days);
  return localIsoDate(date);
}

function reportQuery() {
  const params = new URLSearchParams();
  if ($("reportFrom").value) params.set("from", $("reportFrom").value);
  if ($("reportTo").value) params.set("to", $("reportTo").value);
  if ($("reportOperator").value) params.set("operator_id", $("reportOperator").value);
  return params;
}

function fillReportDates(from, to) {
  $("reportFrom").value = from;
  $("reportTo").value = to;
}

function ensureReportDates() {
  if ($("reportFrom").value && $("reportTo").value) return;
  const today = localIsoDate();
  fillReportDates(shiftIsoDate(today, -6), today);
}

function setReportPreset(kind) {
  const today = localIsoDate();
  if (kind === "today") fillReportDates(today, today);
  else if (kind === "month") fillReportDates(`${today.slice(0, 7)}-01`, today);
  else fillReportDates(shiftIsoDate(today, -6), today);
  loadReport().catch((error) => setStatus(String(error)));
}

function reportBucketLabel(kind, value) {
  if (!value || value === "unknown") return t("unknown");
  if (kind === "country") return countryLabel(value);
  if (kind === "confidence") return t(`conf_${value}`) || value;
  if (kind === "media") {
    if (value === "image") return t("media_image");
    if (value === "video") return t("media_video");
    if (value === "camera") return t("media_camera");
  }
  return value;
}

function tallyTable(rows, labelFn) {
  if (!rows.length) return `<p class="hint">${t("report_empty")}</p>`;
  return `<table class="tally"><thead><tr><th></th><th>${t("col_count")}</th><th>${t("col_share")}</th></tr></thead><tbody>${
    rows.map((row) => {
      const share = Number(row.share || 0);
      return `<tr><td>${labelFn(row)}<i class="bar" style="width:${Math.min(100, share)}%"></i></td><td>${row.count}</td><td>${share.toFixed(1)}%</td></tr>`;
    }).join("")
  }</tbody></table>`;
}

function hourStrip(hours) {
  const max = Math.max(1, ...(hours || []).map((item) => Number(item.count || 0)));
  return `<div class="hour-strip">${(hours || []).map((item) => {
    const count = Number(item.count || 0);
    const height = count ? Math.max(6, Math.round((count / max) * 56)) : 2;
    const hour = String(item.hour).padStart(2, "0");
    return `<div class="hour-cell${count ? " has-count" : ""}"><b>${count || ""}</b><i style="height:${height}px"></i><em>${hour}</em></div>`;
  }).join("")}</div>`;
}

function renderReport(data) {
  state.report = data;
  const errorBox = $("reportError");
  const emptyBox = $("reportEmpty");
  const body = $("reportBody");
  errorBox.hidden = true;
  emptyBox.hidden = true;
  body.innerHTML = "";
  if (data.from) $("reportFrom").value = data.from;
  if (data.to) $("reportTo").value = data.to;
  const select = $("reportOperator");
  const current = select.value;
  select.innerHTML = `<option value="">${t("history_all_operators")}</option>`;
  (data.filter_operators || []).forEach((operator) => {
    const option = document.createElement("option");
    option.value = operator.id == null ? "unassigned" : String(operator.id);
    option.textContent = operatorLabel(operator);
    select.append(option);
  });
  if ([...select.options].some((option) => option.value === current)) select.value = current;
  if (data.database && data.database !== "ok") {
    emptyBox.textContent = data.database === "missing" ? t("report_db_off") : data.database;
    emptyBox.hidden = false;
    return;
  }
  const vehicles = Number(data.vehicle_count || 0);
  const mast = `<div class="report-mast">
      <div><div class="range">${data.from || ""} — ${data.to || ""}</div><span>${t("report_range")}</span></div>
      <div><b>${vehicles}</b><span>${t("report_vehicles")}</span></div>
      <div><b>${data.scan_count || 0}</b><span>${t("report_scans")}</span></div>
      <div><b>${data.operator_count || 0}</b><span>${t("report_operators")}</span></div>
    </div>`;
  if (!vehicles) {
    body.innerHTML = `${mast}<p class="hint">${t("report_empty")}</p>`;
    return;
  }
  const dayRows = (data.days || []).map((row) => ({
    label: row.date,
    count: row.count,
    share: vehicles ? Math.round((Number(row.count || 0) / vehicles) * 1000) / 10 : 0,
  }));
  const operatorRows = (data.operators || []).map((row) => ({
    ...row,
    label: operatorLabel(row),
  }));
  const passages = data.passages || [];
  body.innerHTML = `
    ${mast}
    <div class="report-grid">
      <section class="report-block"><h3>${t("report_by_day")}</h3>${tallyTable(dayRows, (row) => row.label)}</section>
      <section class="report-block"><h3>${t("report_by_hour")}</h3>${hourStrip(data.hours || [])}</section>
      <section class="report-block"><h3>${t("report_by_operator")}</h3>${tallyTable(operatorRows, (row) => row.label)}</section>
      <section class="report-block"><h3>${t("report_by_country")}</h3>${tallyTable(data.countries || [], (row) => reportBucketLabel("country", row.label))}</section>
      <section class="report-block"><h3>${t("report_by_vehicle")}</h3>${tallyTable(data.vehicles || [], (row) => reportBucketLabel("vehicle", row.label))}</section>
      <section class="report-block"><h3>${t("report_by_province")}</h3>${tallyTable(data.provinces || [], (row) => reportBucketLabel("province", row.label))}</section>
      <section class="report-block"><h3>${t("report_by_media")}</h3>${tallyTable(data.media || [], (row) => reportBucketLabel("media", row.label))}</section>
      <section class="report-block"><h3>${t("report_by_confidence")}</h3>${tallyTable(data.confidence || [], (row) => reportBucketLabel("confidence", row.label))}</section>
      <section class="report-block wide">
        <h3>${t("report_passages")}</h3>
        ${passages.length ? `<div class="table-wrap"><table class="passages-table"><thead><tr>
          <th>${t("col_date")}</th><th>${t("col_time")}</th><th>${t("col_plates")}</th>
          <th>${t("history_operator")}</th><th>${t("country")}</th><th>${t("vehicle")}</th><th>${t("province")}</th>
        </tr></thead><tbody>${passages.map((item) => `<tr>
          <td>${item.date || ""}</td><td>${item.scanned_time || ""}</td>
          <td>${[item.plate_prefix, item.plate_number].filter(Boolean).join(" ") || t("plate_empty")}</td>
          <td>${operatorLabel(item)}</td><td>${countryLabel(item.country)}</td>
          <td>${item.vehicle_type || t("unknown")}</td><td>${item.province || t("unknown")}</td>
        </tr>`).join("")}</tbody></table></div>` : `<p class="hint">${t("report_empty")}</p>`}
        ${data.truncated ? `<p class="report-note">${t("report_truncated", { n: passages.length })}</p>` : ""}
      </section>
    </div>
  `;
}

async function loadReport() {
  ensureReportDates();
  const response = await api(`/api/reports?${reportQuery()}`);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    $("reportError").textContent = data.detail || String(response.status);
    $("reportError").hidden = false;
    $("reportEmpty").hidden = true;
    $("reportBody").innerHTML = "";
    return;
  }
  renderReport(data);
}

function openReport() {
  closeCameras();
  closeUsers();
  closeHistory();
  ensureReportDates();
  $("reportPanel").hidden = false;
  $("reportPanel").classList.remove("hidden");
  $("reportError").hidden = true;
  $("reportBtn").setAttribute("aria-pressed", "true");
  loadReport().catch((error) => setStatus(String(error)));
}

function closeReport() {
  $("reportPanel").hidden = true;
  $("reportPanel").classList.add("hidden");
  $("reportBtn").setAttribute("aria-pressed", "false");
}

function downloadReportCsv() {
  ensureReportDates();
  window.location.href = `/api/reports.csv?${reportQuery()}`;
}

function openPassword() {
  $("passwordPanel").hidden = false;
  $("passwordPanel").classList.remove("hidden");
  $("passwordError").hidden = true;
  $("passwordOk").hidden = true;
  $("passwordForm").reset();
}

function closePassword() {
  $("passwordPanel").hidden = true;
  $("passwordPanel").classList.add("hidden");
}

async function savePassword(event) {
  event.preventDefault();
  $("passwordError").hidden = true;
  $("passwordOk").hidden = true;
  const response = await api("/api/auth/password", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      current_password: $("passwordCurrent").value,
      new_password: $("passwordNew").value,
    }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    $("passwordError").textContent = data.detail || t("no_permission");
    $("passwordError").hidden = false;
    return;
  }
  $("passwordOk").textContent = t("password_ok");
  $("passwordOk").hidden = false;
  $("passwordForm").reset();
}

function cancelEditUser() {
  state.editingUserId = null;
  $("userForm").reset();
  $("newUsername").disabled = false;
  $("userCancelEdit").classList.add("hidden");
  $("userCreateBtn").textContent = t("user_create");
  $("newPassword").placeholder = t("placeholder_password");
}

function startEditUser(userId) {
  const user = state.users.find((item) => Number(item.id) === Number(userId));
  if (!user) return;
  state.editingUserId = Number(user.id);
  $("newUsername").value = user.username;
  $("newUsername").disabled = true;
  $("newDisplayName").value = user.display_name || "";
  $("newRole").value = user.role;
  $("newPassword").value = "";
  $("newPassword").placeholder = t("placeholder_password_edit");
  $("userCreateBtn").textContent = t("user_save");
  $("userCancelEdit").classList.remove("hidden");
  $("usersError").hidden = true;
}

async function saveUser(event) {
  event.preventDefault();
  $("usersError").hidden = true;
  const payload = {
    display_name: $("newDisplayName").value,
    role: $("newRole").value,
  };
  const password = $("newPassword").value;
  let response;
  if (state.editingUserId) {
    if (password) payload.password = password;
    response = await api(`/api/users/${state.editingUserId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } else {
    payload.username = $("newUsername").value;
    payload.password = password;
    response = await api("/api/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }
  const data = await response.json();
  if (!response.ok) {
    $("usersError").textContent = data.detail || t("no_permission");
    $("usersError").hidden = false;
    return;
  }
  cancelEditUser();
  await loadUsers();
}

async function deleteUser(userId, name) {
  if (!window.confirm(t("user_delete_confirm", { name: name || userId }))) return;
  const response = await api(`/api/users/${userId}`, { method: "DELETE" });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    $("usersError").textContent = data.detail || t("no_permission");
    $("usersError").hidden = false;
    return;
  }
  if (state.editingUserId === Number(userId)) cancelEditUser();
  await loadUsers();
}

async function toggleUser(userId, currentlyActive) {
  const response = await api(`/api/users/${userId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ active: !currentlyActive }),
  });
  const data = await response.json();
  if (!response.ok) {
    $("usersError").textContent = data.detail || t("no_permission");
    $("usersError").hidden = false;
    return;
  }
  await loadUsers();
}

async function logout() {
  await fetch("/api/auth/logout", { method: "POST" });
  window.location.href = "/login";
}

function bind() {
  document.querySelectorAll(".mode[data-mode]").forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });
  document.querySelectorAll("[data-lang]").forEach((button) => {
    button.addEventListener("click", () => {
      state.lang = button.dataset.lang;
      document.querySelectorAll("[data-lang]").forEach((item) => {
        item.setAttribute("aria-pressed", String(item.dataset.lang === state.lang));
      });
      retranslate();
      loadHealth();
    });
  });
  $("fileInput").addEventListener("change", (event) => acceptFile(event.target.files[0]));
  $("scanBtn").addEventListener("click", () => startScan().catch((error) => setStatus(String(error))));
  $("stopBtn").addEventListener("click", () => stopScan());
  $("saveBtn").addEventListener("click", saveJson);
  $("logoutBtn").addEventListener("click", () => logout().catch((error) => setStatus(String(error))));
  $("usersBtn").addEventListener("click", openUsers);
  $("usersClose").addEventListener("click", closeUsers);
  $("camerasBtn").addEventListener("click", openCameras);
  $("camerasClose").addEventListener("click", closeCameras);
  $("camerasToolbarBtn").addEventListener("click", openCameras);
  $("cameraWallPrev")?.addEventListener("click", () => shiftCameraWall(-1));
  $("cameraWallNext")?.addEventListener("click", () => shiftCameraWall(1));
  $("historyBtn").addEventListener("click", openHistory);
  $("historyClose").addEventListener("click", closeHistory);
  $("historyPrint").addEventListener("click", () => window.print());
  $("reportBtn").addEventListener("click", openReport);
  $("reportClose").addEventListener("click", closeReport);
  $("reportPrint").addEventListener("click", () => window.print());
  $("reportCsv").addEventListener("click", downloadReportCsv);
  $("reportToday").addEventListener("click", () => setReportPreset("today"));
  $("reportWeek").addEventListener("click", () => setReportPreset("week"));
  $("reportMonth").addEventListener("click", () => setReportPreset("month"));
  $("reportFrom").addEventListener("change", () => loadReport().catch((error) => setStatus(String(error))));
  $("reportTo").addEventListener("change", () => loadReport().catch((error) => setStatus(String(error))));
  $("reportOperator").addEventListener("change", () => loadReport().catch((error) => setStatus(String(error))));
  $("historySlipClose").addEventListener("click", closeScanSlip);
  $("historyDate").addEventListener("change", () => loadHistory().catch((error) => setStatus(String(error))));
  $("historyOperator").addEventListener("change", () => loadHistory().catch((error) => setStatus(String(error))));
  $("historyQuery").addEventListener("search", () => loadHistory().catch((error) => setStatus(String(error))));
  $("historyQuery").addEventListener("keydown", (event) => {
    if (event.key !== "Enter") return;
    event.preventDefault();
    loadHistory().catch((error) => setStatus(String(error)));
  });
  $("historyBody").addEventListener("click", (event) => {
    const row = event.target.closest("tr[data-scan-id]");
    if (!row) return;
    openScanSlip(row.dataset.scanId, row.dataset.plateId, row).catch((error) => setStatus(String(error)));
  });
  $("passwordBtn").addEventListener("click", openPassword);
  $("passwordClose").addEventListener("click", closePassword);
  $("passwordForm").addEventListener("submit", (event) => savePassword(event).catch((error) => setStatus(String(error))));
  $("userForm").addEventListener("submit", (event) => saveUser(event).catch((error) => setStatus(String(error))));
  $("userCancelEdit").addEventListener("click", cancelEditUser);
  $("usersBody").addEventListener("click", (event) => {
    const edit = event.target.closest("[data-user-edit]");
    if (edit) {
      startEditUser(Number(edit.dataset.userEdit));
      return;
    }
    const remove = event.target.closest("[data-user-delete]");
    if (remove) {
      deleteUser(Number(remove.dataset.userDelete), remove.dataset.userName).catch((error) => setStatus(String(error)));
      return;
    }
    const button = event.target.closest("[data-user-toggle]");
    if (!button) return;
    toggleUser(Number(button.dataset.userToggle), button.dataset.active === "1").catch((error) => setStatus(String(error)));
  });
  $("roiEnabled").addEventListener("change", () => {
    const enabled = $("roiEnabled").checked;
    if (state.mode === "camera") {
      const host = selectedCameraHost();
      if (!host) return;
      setRoiFor(host, { ...roiFor(host), enabled });
    } else {
      state.roi.enabled = enabled;
      draw();
    }
  });
  $("roiShape").addEventListener("change", () => {
    const shape = $("roiShape").value;
    if (state.mode === "camera") {
      const host = selectedCameraHost();
      if (!host) return;
      setRoiFor(host, { ...roiFor(host), shape, enabled: true });
    } else {
      state.roi.shape = shape;
      draw();
    }
  });
  $("roiFull").addEventListener("click", () => {
    if (state.mode === "camera") {
      const host = selectedCameraHost();
      if (!host) return;
      setRoiFor(host, { ...roiFor(host), enabled: true, x: 0, y: 0, width: 1, height: 1 });
      return;
    }
    state.roi = clampRoi({ ...state.roi, enabled: true, x: 0, y: 0, width: 1, height: 1 });
    $("roiEnabled").checked = true;
    draw();
  });
  $("roiCenter").addEventListener("click", () => {
    if (state.mode === "camera") {
      const host = selectedCameraHost();
      if (!host) return;
      setRoiFor(host, { ...roiFor(host), enabled: true, x: 0.2, y: 0.2, width: 0.6, height: 0.6 });
      return;
    }
    state.roi = clampRoi({ ...state.roi, enabled: true, x: 0.1, y: 0.1, width: 0.8, height: 0.8 });
    $("roiEnabled").checked = true;
    draw();
  });
  $("camStart").addEventListener("click", () => openCamera().catch((error) => setStatus(String(error))));
  $("ipCameraUrl").addEventListener("input", () => {
    if (!state.jobId && state.mode === "camera") $("scanBtn").disabled = !canStartScan();
  });
  $("ipCameraUrl").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addIpCamera().catch((error) => setStatus(String(error)));
    }
  });
  ["ipCameraUser", "ipCameraPassword", "ipCameraPath"].forEach((id) => {
    const field = $(id);
    if (!field) return;
    field.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        addIpCamera().catch((error) => setStatus(String(error)));
      }
    });
  });
  $("ipCameraAddBtn").addEventListener("click", () => addIpCamera().catch((error) => setStatus(String(error))));
  $("ipCameraSaveBtn").addEventListener("click", () => saveIpCamera().catch((error) => setStatus(String(error))));
  if ($("localCameraBtn")) {
    $("localCameraBtn").addEventListener("click", () => addLocalCamera().catch((error) => setStatus(String(error))));
  }
  $("ipCameraRemoveBtn").addEventListener("click", () => removeIpCamera().catch((error) => setStatus(String(error))));
  $("cameraSelectorList")?.addEventListener("change", (event) => {
    const input = event.target.closest("[data-camera-toggle]");
    if (!input) return;
    setCameraSelection(input.dataset.host, input.checked).catch((error) => {
      setStatus(String(error));
      refreshWorker().catch(() => {});
    });
  });
  $("cameraSelectAll")?.addEventListener("click", () => {
    setAllCameraSelections(true).catch((error) => setStatus(String(error)));
  });
  $("cameraClearAll")?.addEventListener("click", () => {
    setAllCameraSelections(false).catch((error) => setStatus(String(error)));
  });
  if ($("computeMode")) {
    $("computeMode").addEventListener("change", () => {
      setComputeMode($("computeMode").value).catch((error) => setStatus(String(error)));
    });
  }
  $("ipCameraSelect").addEventListener("change", () => {
    state.selectedHost = $("ipCameraSelect").value;
    if (state.mode === "camera") $("scanBtn").disabled = !canStartScan();
    const selected = (state.ipCameras || []).find((item) => item.host === $("ipCameraSelect").value);
    $("ipCameraRemoveBtn").disabled = !selected || Boolean(selected.builtin);
    $("ipCameraSaveBtn").disabled = !selected;
    if (selected) {
      $("ipCameraEditName").value = selected.custom_label ? selected.label : "";
      $("ipCameraEditUrl").value = selected.stream_url || "";
    }
    selectCameraHost($("ipCameraSelect").value);
    applyWorker(state.worker || { cameras: [], gpu: {}, plates: [] });
  });
  $("camSnap").addEventListener("click", () => snapshot().catch((error) => setStatus(String(error))));
  $("camRecord").addEventListener("click", () => startRecord().catch((error) => setStatus(String(error))));
  $("camStopRec").addEventListener("click", () => stopRecord().catch((error) => setStatus(String(error))));
  const zone = $("dropZone");
  zone.addEventListener("dragover", (event) => event.preventDefault());
  zone.addEventListener("drop", (event) => {
    event.preventDefault();
    acceptFile(event.dataTransfer.files[0]);
  });
  document.addEventListener("click", (event) => {
    const card = event.target.closest("#crops [data-index], #tableBody [data-index]");
    if (!card) return;
    state.selected = Number(card.dataset.index);
    renderResults();
  });
  document.addEventListener("keydown", (event) => {
    if (!state.plates.length) return;
    const tag = String(event.target?.tagName || "");
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp" && event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const current = state.selected < 0 ? 0 : state.selected;
    const next = (event.key === "ArrowDown" || event.key === "ArrowRight") ? current + 1 : current - 1;
    state.selected = Math.max(0, Math.min(state.plates.length - 1, next));
    renderResults();
  });
  window.addEventListener("resize", () => {
    draw();
    if (state.mode === "camera") layoutCameraWall();
  });
  window.addEventListener("pagehide", () => {
    flushRoiSaves();
    suspendWorkerEvents();
  });
  window.addEventListener("pageshow", (event) => {
    if (event.persisted) resumeWorkerEvents();
  });
  if (window.ResizeObserver && $("cameraGrid")) {
    new ResizeObserver(() => {
      if (state.mode === "camera") layoutCameraWall();
    }).observe($("cameraGrid"));
  }
  bindRoi();
}

bind();
loadMe()
  .then((ok) => {
    if (!ok) return;
    retranslate();
    const first = ["image", "video", "camera"].find((mode) => can(`scan.${mode}`)) || "image";
    setMode(first);
    loadHealth();
    draw();
  })
  .catch((error) => {
    console.error(error);
    setStatus(String(error));
  });
