
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import *
from segment_anything import SamPredictor, sam_model_registry
import torch
import numpy as np
import pandas as pd
import cv2
import sys
import os
import matplotlib.pyplot as plt
import traceback
import datetime
from io import BytesIO

class SAMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Si/SiGe_Selectivity")
        self.setGeometry(100, 100, 1200, 800)
        self.create_menu_bar()
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)  # 여백 제거
        main_layout.setSpacing(0)  # 위젯 간의 간격 제거
        
        # Add banner
        try:
            banner_label = QLabel()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            banner_path = os.path.join(current_dir, "banner.jpg")
            
            if os.path.exists(banner_path):
                banner_pixmap = QPixmap(banner_path)
                scaled_banner = banner_pixmap.scaledToWidth(1200, Qt.SmoothTransformation)
                banner_label.setPixmap(scaled_banner)
                banner_label.setMaximumHeight(scaled_banner.height())
                banner_label.setAlignment(Qt.AlignCenter)  # 중앙 정렬
                main_layout.addWidget(banner_label)
            else:
                print(f"배너 이미지를 찾을 수 없습니다: {banner_path}")
        except Exception as e:
            print(f"배너 로드 중 오류 발생: {str(e)}")

        # Create content widget and layout
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)  # 여백 제거
        content_layout.setSpacing(0)  # 간격 제거
        main_layout.addWidget(content_widget, stretch=1)
        
        # 이미지 뷰어와 컨트롤 패널을 content_layout에 추가
        self.image_view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.image_view.setScene(self.scene)
        content_layout.addWidget(self.image_view, stretch=2)
        
        # 우측 패널 설정
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)  # 우측 패널 여백 제거
        right_layout.setSpacing(0)  # 우측 패널 간격 제거
        content_layout.addWidget(right_panel, stretch=1)
        
        self.setFocusPolicy(Qt.StrongFocus)
        # SAM 모델 초기화
        self.initialize_sam()

        # 분석 파라미터 초기화
        self.measurement_points = 3  # 기본값 3개
        self.pixel_to_nm = 0.1574  # 기존 값
        self.confidence_threshold = 0.5
        self.multimask_output = True
        self.window_size = 100

        self.brush_mode = False
        self.brush_size = 15  # 기본 브러시 크기
        self.brush_add = True  # True: 추가, False: 제거
        self.prev_brush_point = None
        self.brush_points = []
        
        # 외관 설정 초기화
        self.mask_opacity = 50
        self.line_thickness = 2
        self.font_size = 60
        # 변수 초기화
        self.image = None
        self.image_item = None
        self.mask_overlay = None
        self.current_layer = None
        self.layers = []
        self.mode = 'add'
        self.points_add = []
        self.points_remove = []
        self.box_start = None
        self.box_end = None
        self.current_box = None  # 박스 프리뷰용 QGraphicsRectItem
        self.history = []
        self.max_history = 20
        self.box_mode = False  # 박스 모드 상태 추가
        self.show_all_layers = False
        self.layer_colors = [
        [0, 0, 255],    # 빨강
        [0, 255, 0],    # 초록
        [255, 0, 0],    # 파랑
        [0, 255, 255],  # 청록
        [255, 0, 255],  # 마젠타
        [255, 255, 0],  # 노랑
        [0, 0, 128],    # 진한 빨강
        [0, 128, 0],    # 진한 초록
        [128, 0, 0],    # 진한 파랑
    ]
        # 줌 관련 변수 추가
        self.zoom_factor = 1.0

        # UI 설정
        self.setup_ui()
        self.image_view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.image_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.image_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.image_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.image_view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        # 휠 이벤트 연결
        self.image_view.wheelEvent = self.wheel_event

        self.thickness_mode = False
        self.selected_x = None  # X 좌표 저장용

    def create_menu_bar(self):
        # 메뉴바 생성
        menubar = self.menuBar()
            # 메뉴바 스타일 설정
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #317EAA;  /* 짙은 파란색 */
                color: white;  /* 텍스트 색상을 흰색으로 */
            }
            QMenuBar::item {
                spacing: 3px;
                padding: 5px 10px;
                background: transparent;
            }
            QMenuBar::item:selected {
                background: #003B66;  /* 선택 시 더 진한 파란색 */
            }
            QMenuBar::item:pressed {
                background: #003B66;  /* 클릭 시 더 진한 파란색 */
            }
            QMenu {
                background-color: #005B96;  /* 드롭다운 메뉴 배경색 */
                color: white;  /* 드롭다운 메뉴 텍스트 색상 */
                border: 1px solid #003B66;
            }
            QMenu::item:selected {
                background: #003B66;  /* 드롭다운 메뉴 항목 선택 시 색상 */
            }
        """)
        # File 메뉴 생성
        file_menu = menubar.addMenu('File')
        
        # Load Image 액션
        load_action = QAction('Load Image', self)
        load_action.setShortcut('Ctrl+O')
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)
        
        # Save Result 액션
        save_action = QAction('Save Result', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_result)
        file_menu.addAction(save_action)
        
        # 구분선 추가
        file_menu.addSeparator()
        
        # Exit 액션
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Edit 메뉴 생성
        edit_menu = menubar.addMenu('Edit')
        
        # Scale Settings 액션
        scale_action = QAction('Scale Settings', self)
        scale_action.triggered.connect(self.show_scale_dialog)
        edit_menu.addAction(scale_action)
        
        # Analysis Parameters 액션
        analysis_action = QAction('Analysis Parameters', self)
        analysis_action.triggered.connect(self.show_analysis_dialog)
        edit_menu.addAction(analysis_action)
        
        # Appearance Settings 액션
        appearance_action = QAction('Appearance Settings', self)
        appearance_action.triggered.connect(self.show_appearance_dialog)
        edit_menu.addAction(appearance_action)

        # Appearance Settings 액션
        report_action = QAction('Report Settings', self)
        report_action.triggered.connect(self.show_report_dialog)
        edit_menu.addAction(report_action)

        
    def initialize_sam(self):
        try:
            # 절대 경로로 수정
            sam_checkpoint = "C:/Users/Admin/Desktop/김지훈_자료정리/2_Data/8. GB/SAM/segment-anything/segment_anything/sam_vit_h_4b8939.pth"
            model_type = "vit_h"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 모델 파일 존재 여부 확인
            if not os.path.exists(sam_checkpoint):
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {sam_checkpoint}")
                
            self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            self.sam.to(device=device)
            self.predictor = SamPredictor(self.sam)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"SAM 모델 초기화 실패: {str(e)}")
            sys.exit(1)
            
    def setup_ui(self):
        # 우측 패널의 기존 레이아웃 가져오기
        right_panel = self.centralWidget().layout().itemAt(1).widget().layout().itemAt(1).widget()
        right_layout = right_panel.layout()
        
        # 모드 선택
        mode_group = QGroupBox("Mode")
        mode_layout = QVBoxLayout()
        mode_layout.setContentsMargins(5, 5, 5, 5)  # 작은 여백 설정
        self.add_btn = QPushButton("Add")
        self.remove_btn = QPushButton("Remove")
        self.box_btn = QPushButton("Box")
        self.brush_btn = QPushButton("Brush Edit")  # 브러시 편집 버튼
        mode_layout.addWidget(self.add_btn)
        mode_layout.addWidget(self.remove_btn)
        mode_layout.addWidget(self.box_btn)
        mode_layout.addWidget(self.brush_btn)
        mode_group.setLayout(mode_layout)

        # All Layer 토글 버튼 추가
        self.all_layers_btn = QPushButton("All Layers")
        mode_layout.addWidget(self.all_layers_btn)
        
        # 레이어 컨트롤
        layer_group = QGroupBox("Layers")
        layer_layout = QVBoxLayout()
        layer_layout.setContentsMargins(5, 5, 5, 5)  # 작은 여백 설정
        self.layer_list = QListWidget()

        # 레이어 타입 선택 UI 추가
        layer_type_layout = QHBoxLayout()
        layer_type_label = QLabel("Layer Type:")
        self.layer_type_combo = QComboBox()
        self.layer_type_combo.addItems(["Si", "SiGe"])
        self.layer_type_combo.currentIndexChanged.connect(self.change_layer_type)
        layer_type_layout.addWidget(layer_type_label)
        layer_type_layout.addWidget(self.layer_type_combo)

        self.add_layer_btn = QPushButton("New Layer")
        self.delete_layer_btn = QPushButton("Delete Layer")
        layer_layout.addWidget(self.layer_list)
        layer_layout.addLayout(layer_type_layout)
        layer_layout.addWidget(self.add_layer_btn)
        layer_layout.addWidget(self.delete_layer_btn)
        layer_group.setLayout(layer_layout)
        
        # Layer Information 그룹
        info_group = QGroupBox("Layer Information")
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(5, 5, 5, 5)  # 작은 여백 설정
        self.layer_info = QTextEdit()
        self.layer_info.setReadOnly(True)
        info_layout.addWidget(self.layer_info)
        info_group.setLayout(info_layout)
        
        # 우측 패널에 그룹들 추가
        # right_layout.addWidget(file_group)
        right_layout.addWidget(mode_group)
        right_layout.addWidget(layer_group)
        right_layout.addWidget(info_group)
        right_layout.addStretch()

        # Analysis 그룹 추가
        analysis_group = QGroupBox("Analysis")
        analysis_layout = QVBoxLayout()
        analysis_layout.setContentsMargins(5, 5, 5, 5)

        # pixel to nm 설정 버튼
        self.reset_btn = QPushButton("Reset")
        self.thickness_btn = QPushButton("Selectivity Calculate")  # 버튼 텍스트 변경
        self.adjust_points_btn = QPushButton("Adjust Point")
        
        analysis_layout.addWidget(self.reset_btn)
        analysis_layout.addWidget(self.adjust_points_btn) 
        analysis_layout.addWidget(self.thickness_btn)
        analysis_group.setLayout(analysis_layout)
        
        # 우측 패널에 Analysis 그룹 추가
        right_layout.addWidget(analysis_group)
        right_layout.addStretch()

        # 브러시 설정 그룹 추가
        brush_settings_group = QGroupBox("Brush Settings")
        brush_settings_layout = QVBoxLayout()
        brush_settings_layout.setContentsMargins(5, 5, 5, 5)
        
        # 브러시 크기 설정
        brush_size_layout = QHBoxLayout()
        brush_size_label = QLabel("Size:")
        self.brush_size_spinner = QSpinBox()
        self.brush_size_spinner.setRange(1, 100)
        self.brush_size_spinner.setValue(self.brush_size)
        self.brush_size_spinner.valueChanged.connect(self.set_brush_size)
        brush_size_layout.addWidget(brush_size_label)
        brush_size_layout.addWidget(self.brush_size_spinner)
        
        # 브러시 타입 설정
        brush_type_layout = QHBoxLayout()
        self.brush_add_radio = QRadioButton("Add")
        self.brush_remove_radio = QRadioButton("Remove")
        self.brush_add_radio.setChecked(True)
        self.brush_add_radio.toggled.connect(self.set_brush_type)
        brush_type_layout.addWidget(self.brush_add_radio)
        brush_type_layout.addWidget(self.brush_remove_radio)
        
        brush_settings_layout.addLayout(brush_size_layout)
        brush_settings_layout.addLayout(brush_type_layout)
        brush_settings_group.setLayout(brush_settings_layout)
        
        # 우측 패널에 브러시 설정 그룹 추가
        right_layout.addWidget(brush_settings_group)
        brush_settings_group.setVisible(False)  # 초기에는 숨김
        self.brush_settings_group = brush_settings_group  # 참조 저장

        # 이벤트 연결
        self.connect_events()

    def change_layer_type(self):
        """현재 선택된 레이어의 타입 변경"""
        if self.current_layer is None or self.current_layer >= len(self.layers):
            return
            
        layer_type = self.layer_type_combo.currentText()
        self.layers[self.current_layer]['layer_type'] = layer_type
        self.update_layer_info()  # 레이어 정보 업데이트

    def set_pixel_scale(self):
        value, ok = QInputDialog.getDouble(
            self, 
            "Set Scale",
            "Enter pixel to nm ratio:",
            self.pixel_to_nm,
            0.0001, 1000.0, 4
        )
        if ok:
            self.pixel_to_nm = value

    def measure_thickness_at_x(self, x):
        """선택된 x 좌표에서 두께 측정"""
        if self.current_layer is None or len(self.layers[self.current_layer]['masks']) == 0:
            return
            
        current = self.layers[self.current_layer]
        mask = current['masks'][0].astype(np.uint8) * 255
        
        # y 좌표 찾기 (아랫면)
        y_coords = np.where(mask[:, x] > 0)[0]
        if len(y_coords) == 0:
            return
            
        y_bottom = np.max(y_coords)
        
        # 접선 계산을 위한 설정
        window_size = 150
        nearby_points_x = []
        nearby_points_y = []
        
        # 주변 점들 수집 (접선 계산용)
        for dx in range(-window_size, window_size+1):
            curr_x = x + dx
            if 0 <= curr_x < mask.shape[1]:
                curr_y_coords = np.where(mask[:, curr_x] > 0)[0]
                if len(curr_y_coords) > 0:
                    curr_y = np.max(curr_y_coords)
                    nearby_points_x.append(curr_x)
                    nearby_points_y.append(curr_y)
        
        if len(nearby_points_x) > 2:
            # 선형 회귀로 접선의 기울기 계산
            coeffs = np.polyfit(nearby_points_x, nearby_points_y, 1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # 접선 그리기용 점 계산
            x1 = x - window_size
            x2 = x + window_size
            y1 = int(slope * x1 + intercept)
            y2 = int(slope * x2 + intercept)
            
            # 수직선 관련 계산
            if abs(slope) > 1e-6:  # 기울기가 0이 아닌 경우
                perpendicular_slope = -1/slope
                perpendicular_length = mask.shape[0]
                
                # 교차점 찾기
                intersection_found = False
                last_valid_point = None
                first_exit_point = None
                
                for t in range(0, abs(perpendicular_length)):
                    if perpendicular_slope > 0:
                        check_x = int(x - t/np.sqrt(1 + perpendicular_slope**2))
                        check_y = int(y_bottom - perpendicular_slope * t/np.sqrt(1 + perpendicular_slope**2))
                    else:
                        check_x = int(x + t/np.sqrt(1 + perpendicular_slope**2))
                        check_y = int(y_bottom + perpendicular_slope * t/np.sqrt(1 + perpendicular_slope**2))
                    
                    # 이미지 범위 체크
                    if (check_x < 0 or check_x >= mask.shape[1] or 
                        check_y < 0 or check_y >= mask.shape[0]):
                        break
                    
                    # 마스크 내부의 마지막 점 저장
                    if mask[check_y, check_x] > 0:
                        last_valid_point = (check_x, check_y)
                    # 마스크를 벗어난 첫 점 저장
                    elif last_valid_point is not None and first_exit_point is None:
                        first_exit_point = (check_x, check_y)
                        
                        # 윗면 근처인지 확인
                        y_vals = np.where(mask[:, check_x] > 0)[0]
                        if len(y_vals) > 0:
                            y_top = np.min(y_vals)
                            if abs(check_y - y_top) < 10:  # 윗면과 가까운 경우
                                intersection_found = True
                                distance = np.sqrt((check_x - x)**2 + (y_top - y_bottom)**2)
                                distance_nm = distance * self.pixel_to_nm
                                
                                # 측정 정보 저장 (한 번만)
                                measurement_info = {
                                    'start_point': [x, y_bottom],
                                    'end_point': [check_x, y_top],
                                    'distance': distance_nm,
                                    'tangent_points': [(x1, y1), (x2, y2)],
                                    'slope': slope
                                }
                                
                                if 'thickness_measurements' not in current:
                                    current['thickness_measurements'] = []
                                    current['thickness_points'] = []
                                    current['thickness_values'] = []
                                
                                current['thickness_measurements'].append(measurement_info)
                                current['thickness_points'].append([x, y_bottom])
                                current['thickness_values'].append(distance_nm)
                                
                                # 측정 결과 표시 (한 번만)
                                self.draw_thickness_measurement(x, y_bottom, check_x, y_top, 
                                                            distance_nm, [(x1, y1), (x2, y2)], slope)
                                break
                
                if not intersection_found and last_valid_point is not None:
                    distance = np.sqrt((last_valid_point[0] - x)**2 + 
                                    (last_valid_point[1] - y_bottom)**2)
                    distance_nm = distance * self.pixel_to_nm
                    
                    # thickness 관련 리스트가 없으면 초기화
                    if 'thickness_measurements' not in current:
                        current['thickness_measurements'] = []
                        current['thickness_points'] = []
                        current['thickness_values'] = []
                    
                    # 이미 존재하는 측정인지 확인
                    is_duplicate = False
                    for existing_point in current['thickness_points']:
                        if existing_point[0] == x and existing_point[1] == y_bottom:
                            is_duplicate = True
                            break
                    
                    # 중복이 아닌 경우에만 저장 및 그리기
                    if not is_duplicate:
                        measurement_info = {
                            'start_point': [x, y_bottom],
                            'end_point': [last_valid_point[0], last_valid_point[1]],
                            'distance': distance_nm,
                            'tangent_points': [(x1, y1), (x2, y2)],
                            'slope': slope
                        }
                        
                        current['thickness_measurements'].append(measurement_info)
                        current['thickness_points'].append([x, y_bottom])
                        current['thickness_values'].append(distance_nm)
                        
                        # 새로운 측정점만 그리기
                        self.draw_thickness_measurement(x, y_bottom, last_valid_point[0], last_valid_point[1],
                                                    distance_nm, [(x1, y1), (x2, y2)], slope)
                        
                        # Layer Information만 업데이트
                        self.update_layer_info()

    def calculate_average_thickness(self, mask):
        """마스크의 평균 두께를 계산"""
        try:
            mask_binary = mask.astype(np.uint8) * 255
            valid_thicknesses = []
            measurement_points = []
            
            # x 좌표 범위 설정 (가장자리 20픽셀 제외)
            x_points = range(20, mask_binary.shape[1]-20)
            
            for x in x_points:
                # y 좌표 찾기 (아랫면)
                y_coords = np.where(mask_binary[:, x] > 0)[0]
                if len(y_coords) == 0:
                    continue
                    
                y_bottom = np.max(y_coords)
                
                # 접선 계산을 위한 설정
                window_size = 100  # measure_thickness_at_x와 동일한 윈도우 크기
                nearby_points_x = []
                nearby_points_y = []
                
                # 주변 점들 수집 (접선 계산용)
                for dx in range(-window_size, window_size+1):
                    curr_x = x + dx
                    if 0 <= curr_x < mask_binary.shape[1]:
                        curr_y_coords = np.where(mask_binary[:, curr_x] > 0)[0]
                        if len(curr_y_coords) > 0:
                            curr_y = np.max(curr_y_coords)
                            nearby_points_x.append(curr_x)
                            nearby_points_y.append(curr_y)
                            
                if len(nearby_points_x) > 2:
                    # 선형 회귀로 접선의 기울기 계산
                    coeffs = np.polyfit(nearby_points_x, nearby_points_y, 1)
                    slope = coeffs[0]
                    
                    # 수직선 관련 계산
                    if abs(slope) > 1e-6:  # 기울기가 0이 아닌 경우
                        perpendicular_slope = -1/slope
                        perpendicular_length = mask_binary.shape[0]
                        
                        # 상단 교차점 찾기
                        for t in range(0, abs(perpendicular_length)):
                            if perpendicular_slope > 0:
                                check_x = int(x - t/np.sqrt(1 + perpendicular_slope**2))
                                check_y = int(y_bottom - perpendicular_slope * t/np.sqrt(1 + perpendicular_slope**2))
                            else:
                                check_x = int(x + t/np.sqrt(1 + perpendicular_slope**2))
                                check_y = int(y_bottom + perpendicular_slope * t/np.sqrt(1 + perpendicular_slope**2))
                            
                            # 이미지 범위 체크
                            if (check_x < 0 or check_x >= mask_binary.shape[1] or 
                                check_y < 0 or check_y >= mask_binary.shape[0]):
                                break
                            
                            # 윗면 근처인지 확인
                            y_vals = np.where(mask_binary[:, check_x] > 0)[0]
                            if len(y_vals) > 0:
                                y_top = np.min(y_vals)
                                if abs(check_y - y_top) < 10:  # 윗면과 가까운 경우
                                    distance = np.sqrt((check_x - x)**2 + (y_top - y_bottom)**2)
                                    distance_nm = distance * self.pixel_to_nm
                                    valid_thicknesses.append(distance_nm)
                                    measurement_points.append((x, y_bottom, check_x, y_top))
                                    break
                                
            if valid_thicknesses:
                # 그래프 생성
                fig = plt.figure(figsize=(10, 3))
                
                plt.plot(pd.Series(valid_thicknesses).rolling(window=10).mean(), 'gray')
                plt.grid(True, alpha=0.3)
                # plt.legend(fontsize=10)
                plt.tight_layout()
                
                # 이미지를 BytesIO에 저장
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)

                return {
                    'average': np.mean(valid_thicknesses),
                    'std': np.std(valid_thicknesses),
                    'min': np.min(valid_thicknesses),
                    'max': np.max(valid_thicknesses),
                    'count': len(valid_thicknesses),
                    'points': measurement_points,
                    'thickness' :valid_thicknesses,
                    'plot': buf
                }
            else:
                return None
                
        except Exception as e:
            print(f"Error in thickness calculation: {str(e)}")
            traceback.print_exc()
            return None

    def draw_thickness_measurement(self, x1, y1, x2, y2, distance_nm, tangent_points, slope):
        """두께 측정 결과를 화면에 표시"""
        try:
            # 시작점과 끝점 표시
            pen = QPen(Qt.green)
            pen.setWidth(self.line_thickness)
            self.scene.addEllipse(x1-5, y1-5, 10, 10, pen)
            self.scene.addEllipse(x2-5, y2-5, 10, 10, pen)
            
            # 접선 그리기
            t1, t2 = tangent_points
            self.scene.addLine(t1[0], t1[1], t2[0], t2[1], pen)
            
            # 측정선 그리기
            pen.setWidth(self.line_thickness+1)
            self.scene.addLine(x1, y1, x2, y2, pen)
            
            # 현재 측정 번호 계산 (measurements 배열의 길이 사용)
            current_layer = self.layers[self.current_layer]
            # point_number = len(current_layer['thickness_measurements'])
            # text = f"P{point_number}: {distance_nm:.2f}nm"
            
            # 새로운 포인트의 번호 결정 
            point_number = 1  # 기본값
            existing_points = []
            
            if 'thickness_measurements' in current_layer:
                for i, measurement in enumerate(current_layer['thickness_measurements']):
                    if measurement['start_point'][0] == x1 and measurement['start_point'][1] == y1:
                        point_number = i + 1  # 기존 포인트면 해당 번호 사용
                        break
                    existing_points.append((measurement['start_point'][0], measurement['start_point'][1]))
                
                if (x1, y1) not in existing_points:
                    point_number = len(existing_points) + 1  # 새로운 포인트면 다음 번호 사용

            text = f"P{point_number}: {distance_nm:.2f}nm"



            # 텍스트 아이템 생성 및 설정
            text_item = self.scene.addText(text)
            text_item.setDefaultTextColor(Qt.yellow)
            font = QFont()
            font.setPointSize(self.font_size)
            text_item.setFont(font)

            # 텍스트 위치 계산
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            text_item.setPos(mid_x + 20, mid_y)
            
            # 텍스트 배경 추가
            text_rect = text_item.boundingRect()
            text_rect.translate(mid_x + 20, mid_y)
            background = self.scene.addRect(
                text_rect,
                QPen(Qt.NoPen),
                QBrush(QColor(0, 0, 0, 128))
            )
            background.setZValue(text_item.zValue() - 1)
            
        except Exception as e:
            print(f"Error drawing measurement: {str(e)}")
            traceback.print_exc()
            
    def update_layer_info(self):
        info_text = ""
        for i, layer in enumerate(self.layers):
            layer_type = layer.get('layer_type', 'Si')
            info_text += f"Layer {i+1} ({layer_type}):\n"
            if len(layer['masks']) > 0:
                size = layer['sizes'][0]
                info_text += f"  - Pixel Count: {size[0]:,}\n"
                info_text += f"  - Coverage: {size[1]:.2f}%\n"
                info_text += f"  - Points: {len(layer['points_add'])} add, {len(layer['points_remove'])} remove\n"
                
                # 두께 측정 정보 추가
                if 'thickness_measurements' in layer and len(layer['thickness_points']) > 0:
                    info_text += "\nThickness Measurements:\n"
                    for j, (point, value) in enumerate(zip(layer['thickness_points'], layer['thickness_values'])):
                        info_text += f"  Point {j+1}: x={point[0]}, thickness={value:.2f}nm\n"
            else:
                info_text += "  - No mask\n"
            info_text += "\n"
        
        self.layer_info.setText(info_text)

    def wheel_event(self, event):
        if event.modifiers() == Qt.ControlModifier:
            # Ctrl + 휠 이벤트 처리
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in(event.pos())
            else:
                self.zoom_out(event.pos())
            event.accept()
        else:
            # 일반 스크롤은 기본 동작 유지
            super(QGraphicsView, self.image_view).wheelEvent(event)

    def zoom_in(self, pos):
        self.zoom_factor = min(self.zoom_factor * 1.1, 10.0)
        self.update_zoom(pos)

    def zoom_out(self, pos):
        self.zoom_factor = max(self.zoom_factor / 1.1, 0.1)
        self.update_zoom(pos)

    def update_zoom(self, pos):
        old_pos = self.image_view.mapToScene(pos)
        self.image_view.setTransform(QTransform().scale(self.zoom_factor, self.zoom_factor))
        new_pos = self.image_view.mapToScene(pos)
        delta = new_pos - old_pos
        self.image_view.horizontalScrollBar().setValue(
            self.image_view.horizontalScrollBar().value() + delta.x())
        self.image_view.verticalScrollBar().setValue(
            self.image_view.verticalScrollBar().value() + delta.y())
        
    def connect_events(self):
        # 파일 관련
        # self.load_btn.clicked.connect(self.load_image)
        # self.save_btn.clicked.connect(self.save_result)
        
        # 모드 관련
        self.add_btn.clicked.connect(lambda: self.set_mode('add'))
        self.remove_btn.clicked.connect(lambda: self.set_mode('remove'))
        self.brush_btn.clicked.connect(self.brush_mode_edit)  # 브러시 편집 다이얼로그 호출
        
        # 레이어 관련
        self.add_layer_btn.clicked.connect(self.add_new_layer)
        self.delete_layer_btn.clicked.connect(self.delete_current_layer)
        self.layer_list.currentRowChanged.connect(self.change_layer)
        
        # 마우스 이벤트
        self.image_view.mousePressEvent = self.mouse_press_event
        self.image_view.mouseMoveEvent = self.mouse_move_event
        self.image_view.mouseReleaseEvent = self.mouse_release_event
        self.all_layers_btn.clicked.connect(self.toggle_all_layers)
        self.reset_btn.clicked.connect(self.reset)
        self.adjust_points_btn.clicked.connect(self.adjust_layer_points)
        self.thickness_btn.clicked.connect(self.calculate_selectivity)  # 자동 측정 함수 연결


    def reset(self):
        """모든 측정 데이터와 레이어를 초기 상태로 초기화하는 함수"""
        try:
            # Selectivity 측정 데이터 초기화
            if hasattr(self, 'selectivity_data'):
                del self.selectivity_data
                del self.adjusted_si_endpoints
                del self.adjusted_sige_centers

            # 레이어 리스트 초기화
            self.layers = []
            self.current_layer = None

            self.initialize_sam()
            
            # 레이어 리스트 위젯 초기화
            self.layer_list.clear()
            
            # 포인트 데이터 초기화
            self.points_add = []
            self.points_remove = []
            self.box_start = None
            self.box_end = None
            
            # 모드 초기화
            self.mode = 'add'
            self.thickness_mode = False
            self.box_mode = False
            self.show_all_layers = False
            
            # 선택된 좌표 초기화
            self.selected_x = None
            
            # 줌 팩터 초기화
            self.zoom_factor = 1.0
            
            # 씬 초기화
            if hasattr(self, 'scene'):
                self.scene.clear()
                if self.image is not None:
                    # 원본 이미지 다시 표시
                    height, width = self.image.shape[:2]
                    bytes_per_line = 3 * width
                    q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    self.image_item = self.scene.addPixmap(pixmap)
            
            # 첫 레이어 생성
            self.add_new_layer()
            
            # 레이어 정보 업데이트
            self.update_layer_info()
            
            # 버튼 상태 초기화
            self.add_btn.setChecked(True)
            self.remove_btn.setChecked(False)
            self.box_btn.setChecked(False)
            self.all_layers_btn.setChecked(False)
            
            print("Reset completed successfully - all layers and measurements initialized")
            
        except Exception as e:
            print(f"Error during reset: {str(e)}")
            traceback.print_exc()

    def toggle_all_layers(self):
        self.show_all_layers = not self.show_all_layers
        self.display_image()

    def mouse_press_event(self, event):
        if self.image is None:
            return
            
        scene_pos = self.image_view.mapToScene(event.pos())
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        
        print(f"Mouse clicked at: ({x}, {y})")  # 디버깅
        print(f"Current mode - thickness: {self.thickness_mode}")  # 디버깅
        print(f"Current layer: {self.current_layer}")  # 디버깅
        
        # 이미지 범위 체크
        if x < 0 or x >= self.image.shape[1] or y < 0 or y >= self.image.shape[0]:
            print("Click outside image bounds")  # 디버깅
            return
            
        # 현재 레이어 체크
        if self.current_layer is None or self.current_layer >= len(self.layers):
            QMessageBox.warning(self, "Warning", "활성화된 레이어가 없습니다.")
            return

        if self.brush_mode:
            # 브러시 모드에서는 상태 저장 및 초기화
            self.save_state()
            self.prev_brush_point = (x, y)
            self.brush_points = []
            self.apply_brush(x, y) 

        elif self.thickness_mode:
            try:
                print("Attempting thickness measurement...")  # 디버깅
                if len(self.layers[self.current_layer]['masks']) > 0:
                    print(f"Mask shape: {self.layers[self.current_layer]['masks'][0].shape}")  # 디버깅
                    self.measure_thickness_at_x(x)
                else:
                    print("No mask in current layer")  # 디버깅
            except Exception as e:
                print(f"Error in thickness measurement: {str(e)}")  # 디버깅
                traceback.print_exc()  # 상세한 에러 정보 출력
        elif self.box_mode:
            print("Box mode - setting start point")  # 디버깅
            self.box_start = (x, y)
            self.save_state()
        else:
            print("Add/Remove mode - adding point")  # 디버깅
            self.add_point(x, y)

    def mouse_move_event(self, event):
        """마우스 이동 이벤트 처리"""
        if self.box_mode and self.box_start:
            pos = self.image_view.mapToScene(event.pos())
            self.box_end = (int(pos.x()), int(pos.y()))
            self.update_box_preview()
        elif self.brush_mode and self.prev_brush_point is not None:
            # 브러시 모드에서 마우스 이동 처리
            pos = self.image_view.mapToScene(event.pos())
            x = int(pos.x())
            y = int(pos.y())
            
            # 이미지 범위 확인
            if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                # 이전 점과 현재 점 사이를 보간하여 브러시 적용
                if self.prev_brush_point:
                    prev_x, prev_y = self.prev_brush_point
                    dx = x - prev_x
                    dy = y - prev_y
                    
                    # 두 점 사이의 거리
                    distance = max(1, int(((dx ** 2) + (dy ** 2)) ** 0.5))
                    
                    # 두 점 사이를 일정 간격으로 나누어 브러시 적용
                    for i in range(1, distance + 1):
                        ix = int(prev_x + dx * i / distance)
                        iy = int(prev_y + dy * i / distance)
                        
                        # 브러시 적용
                        self.apply_brush(ix, iy)
                
                self.prev_brush_point = (x, y)

    def mouse_release_event(self, event):
        """마우스 릴리즈 이벤트 처리"""
        if self.box_mode and self.box_start:
            pos = self.image_view.mapToScene(event.pos())
            self.box_end = (int(pos.x()), int(pos.y()))
            
            if self.box_end != self.box_start:
                self.process_box()
            
            # 프리뷰 박스 제거 및 상태 초기화
            if hasattr(self, 'current_box') and self.current_box is not None:
                try:
                    self.scene.removeItem(self.current_box)
                except:
                    pass
            self.current_box = None
            self.box_start = None
            self.box_end = None
        elif self.brush_mode:
            # 브러시 모드에서 마우스 릴리즈 처리
            self.prev_brush_point = None
            
            # 브러시로 수집된 포인트가 있다면 마스크 업데이트
            if self.brush_points:
                self.update_mask_from_brush()


    def apply_brush(self, x, y):
        """브러시를 적용하여 마스크를 수정"""
        if self.current_layer is None or self.current_layer >= len(self.layers):
            return
            
        current = self.layers[self.current_layer]
        
        # 마스크가 없으면 초기화
        if 'masks' not in current or len(current['masks']) == 0:
            height, width = self.image.shape[:2]
            current['masks'] = [np.zeros((height, width), dtype=bool)]
            current['sizes'] = [(0, 0.0)]
        
        # 현재 마스크 가져오기
        mask = current['masks'][0].copy()
        
        # 브러시 영역 계산
        y_coords, x_coords = np.ogrid[-self.brush_size:self.brush_size+1, -self.brush_size:self.brush_size+1]
        brush_mask = x_coords**2 + y_coords**2 <= self.brush_size**2
        
        # 브러시 적용 범위 계산
        y_min = max(0, y - self.brush_size)
        y_max = min(mask.shape[0], y + self.brush_size + 1)
        x_min = max(0, x - self.brush_size)
        x_max = min(mask.shape[1], x + self.brush_size + 1)
        
        # 브러시 마스크 부분
        brush_y_min = max(0, self.brush_size - y)
        brush_y_max = brush_y_min + (y_max - y_min)
        brush_x_min = max(0, self.brush_size - x)
        brush_x_max = brush_x_min + (x_max - x_min)
        brush_part = brush_mask[brush_y_min:brush_y_max, brush_x_min:brush_x_max]
        
        # 마스크 수정 (추가 또는 제거)
        if self.brush_add:
            mask[y_min:y_max, x_min:x_max][brush_part] = True
        else:
            mask[y_min:y_max, x_min:x_max][brush_part] = False
        
        # 마스크 업데이트
        current['masks'][0] = mask
        
        # 브러시 포인트 저장 (나중에 SAM 모델에 전달할 수 있도록)
        self.brush_points.append((x, y, self.brush_add))
        
        # 픽셀 수와 퍼센트 계산
        pixel_count, percentage = self.calculate_mask_size(mask)
        current['sizes'][0] = (pixel_count, percentage)
        
        # 화면 업데이트
        self.display_image()

    def update_mask_from_brush(self):
        """브러시로 수정된 마스크를 SAM 모델에 반영"""
        if not self.brush_points:
            return
            
        current = self.layers[self.current_layer]
        
        # 브러시 포인트를 SAM 모델에 전달할 포인트로 변환
        for x, y, is_add in self.brush_points:
            if is_add:
                current['points_add'].append([x, y])
            else:
                current['points_remove'].append([x, y])
        
        # SAM 모델로 마스크 업데이트
        self.update_mask()
        
        # 브러시 포인트 초기화
        self.brush_points = []

    def add_point(self, x, y):
        if self.current_layer is None or self.current_layer >= len(self.layers):
            QMessageBox.warning(self, "Warning", "활성화된 레이어가 없습니다.")
            return
                
        self.save_state()  # 상태 저장
        current = self.layers[self.current_layer]
        
        if self.mode == 'add':
            current['points_add'].append([x, y])
        elif self.mode == 'remove':
            current['points_remove'].append([x, y])
        
        # 바로 마스크 업데이트
        self.update_mask()


    def update_box_preview(self):
        """박스 프리뷰 업데이트"""
        try:
            # 이전 박스 제거
            if hasattr(self, 'current_box') and self.current_box is not None:
                try:
                    self.scene.removeItem(self.current_box)
                except:
                    pass
                self.current_box = None
            
            if self.box_start and self.box_end:
                x1, y1 = self.box_start
                x2, y2 = self.box_end
                
                # 박스 좌표 계산
                rect_x = min(x1, x2)
                rect_y = min(y1, y2)
                rect_width = abs(x2 - x1)
                rect_height = abs(y2 - y1)
                
                # 새 박스 생성
                pen = QPen(Qt.red)
                pen.setStyle(Qt.DashLine)
                pen.setWidth(2)
                self.current_box = self.scene.addRect(
                    rect_x, rect_y, rect_width, rect_height, 
                    pen
                )
                
        except Exception as e:
            print(f"박스 프리뷰 업데이트 중 오류 발생: {str(e)}")

    def process_box(self):
        if not self.box_start or not self.box_end:
            return

        x1, y1 = self.box_start
        x2, y2 = self.box_end

        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        box = np.array([x_min, y_min, x_max, y_max])

        current = self.layers[self.current_layer]
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],
            multimask_output=self.multimask_output
        )

        if len(masks) > 0:
            best_mask_idx = scores.argmax()
            current['masks'] = [masks[best_mask_idx]]
            pixel_count, percentage = self.calculate_mask_size(masks[best_mask_idx])
            current['sizes'] = [(pixel_count, percentage)]

        self.display_image()
        
    def load_image(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
        if file_path:
            try:
                self.current_image_path = file_path
                
                image = np.fromfile(file_path, np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                if image is None:
                    raise Exception("이미지를 불러올 수 없습니다.")
                
                # BGR to RGB 변환
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # 이미지 설정
                self.image = image
                self.predictor.set_image(image)
                
                # 첫 레이어 생성
                if not self.layers:
                    self.add_new_layer()
                
                # 이미지 표시
                self.display_image()
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"이미지 로드 실패: {str(e)}")

    def save_state(self):
        """현재 상태를 히스토리에 저장"""
        try:
            current_state = {
                'layers': [layer.copy() for layer in self.layers],
                'current_layer': self.current_layer
            }
            self.history.append(current_state)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            print(f"State saved, history size: {len(self.history)}")  # 디버깅용
        except Exception as e:
            print(f"Error saving state: {str(e)}")  # 디버깅용

    def undo(self):
        if len(self.history) > 0:
            try:
                previous_state = self.history.pop()
                print(f"Restoring state, history size: {len(self.history)}")  # 디버깅용
                
                self.layers = [layer.copy() for layer in previous_state['layers']]
                self.current_layer = previous_state['current_layer']
                
                # 레이어 리스트 업데이트
                self.layer_list.clear()
                for layer in self.layers:
                    self.layer_list.addItem(layer['name'])
                self.layer_list.setCurrentRow(self.current_layer)
                
                # 화면 갱신
                self.display_image()
            except Exception as e:
                print(f"Error during undo: {str(e)}")  # 디버깅용

    def display_image(self):
        if self.image is None:
            return
                
        # 기존 아이템들 제거
        self.scene.clear()
        
        # 이미지 표시
        height, width = self.image.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.image_item = self.scene.addPixmap(pixmap)
        
        if self.show_all_layers:
            # 모든 레이어의 마스크 표시
            for layer_idx, layer in enumerate(self.layers):
                if 'masks' in layer and len(layer['masks']) > 0:
                    mask = layer['masks'][0]
                    
                    # 마스크 오버레이 생성
                    mask_image = np.zeros((height, width, 4), dtype=np.uint8)
                    color = np.concatenate([np.random.random(3) * 255, [128]])  # 랜덤 색상, 50% 투명도
                    mask_image[mask] = color
                    
                    # 마스크 표시
                    mask_qimg = QImage(mask_image.data, width, height, QImage.Format_RGBA8888)
                    mask_pixmap = QPixmap.fromImage(mask_qimg)
                    self.scene.addPixmap(mask_pixmap)
                    
                    # # 마스크 크기 정보 표시
                    # if 'sizes' in layer and len(layer['sizes']) > 0:
                    #     pixel_count, percentage = layer['sizes'][0]
                        
                    #     # 마스크 중심점 계산
                    #     y, x = np.where(mask)
                    #     if len(y) > 0 and len(x) > 0:
                    #         center_y = int(np.mean(y))
                    #         center_x = int(np.mean(x))
                            
                    #         # 텍스트 생성
                    #         label = f'L{layer_idx+1}: {pixel_count:,} px\n{percentage:.1f}%'
                            
                    #         # 텍스트 배경 생성
                    #         text_item = self.scene.addText(label)
                    #         text_item.setDefaultTextColor(Qt.white)
                            
                    #         # 텍스트 배경 추가
                    #         text_rect = text_item.boundingRect()
                    #         background = self.scene.addRect(
                    #             text_rect,
                    #             QPen(Qt.NoPen),
                    #             QBrush(QColor(0, 0, 0, 128))
                    #         )
                            
                    #         # 텍스트 위치 조정
                    #         text_item.setPos(
                    #             center_x - text_rect.width()/2,
                    #             center_y - text_rect.height()/2
                    #         )
                    #         background.setPos(
                    #             center_x - text_rect.width()/2,
                    #             center_y - text_rect.height()/2
                    #         )
        else:
            # 현재 레이어의 마스크만 표시
            if self.current_layer is not None and 0 <= self.current_layer < len(self.layers) and not hasattr(self, 'selectivity_data'):
                current = self.layers[self.current_layer]
                if 'masks' in current and len(current['masks']) > 0:
                    mask = current['masks'][0]
                    
                    # 마스크 오버레이 생성
                    mask_image = np.zeros((height, width, 4), dtype=np.uint8)
                    mask_image[mask] = [255, 0, 0, int(self.mask_opacity * 255 / 100)]  # opacity 적용
                    
                    # 마스크 표시
                    mask_qimg = QImage(mask_image.data, width, height, QImage.Format_RGBA8888)
                    mask_pixmap = QPixmap.fromImage(mask_qimg)
                    self.scene.addPixmap(mask_pixmap)
                    
                    # # 마스크 크기 정보 표시
                    # if 'sizes' in current and len(current['sizes']) > 0:
                    #     pixel_count, percentage = current['sizes'][0]
                        
                    #     # 마스크 중심점 계산
                    #     y, x = np.where(mask)
                    #     if len(y) > 0 and len(x) > 0:
                    #         center_y = int(np.mean(y))
                    #         center_x = int(np.mean(x))
                            
                    #         # 텍스트 생성
                    #         label = f'{pixel_count:,} px\n{percentage:.1f}%'
                            
                    #         # 텍스트 항목 생성
                    #         text_item = self.scene.addText(label)
                    #         text_item.setDefaultTextColor(Qt.white)
                            
                    #         # 텍스트 배경 추가
                    #         text_rect = text_item.boundingRect()
                    #         background = self.scene.addRect(
                    #             text_rect,
                    #             QPen(Qt.NoPen),
                    #             QBrush(QColor(0, 0, 0, 128))
                    #         )
                            
                    #         # 텍스트 위치 조정
                    #         text_item.setPos(
                    #             center_x - text_rect.width()/2,
                    #             center_y - text_rect.height()/2
                    #         )
                    #         background.setPos(
                    #             center_x - text_rect.width()/2,
                    #             center_y - text_rect.height()/2
                    #         )

        # Selectivity 계산 결과 표시
        if hasattr(self, 'selectivity_data'):
            # Si 레이어 두께 측정 표시
            for measurement in self.selectivity_data['post_si_thickness_measurements']:
                if 'start_point' in measurement and 'end_point' in measurement:
                    start_x, start_y = measurement['start_point']
                    end_x, end_y = measurement['end_point']
                    
                    # 측정선 그리기
                    pen = QPen(Qt.yellow)
                    pen.setWidth(self.line_thickness)
                    self.scene.addLine(start_x, start_y, end_x, end_y, pen)
                    
                    # 두께 텍스트
                    if 'thickness_nm' in measurement:
                        text = self.scene.addText(f"{measurement['thickness_nm']:.2f}nm")
                        text.setDefaultTextColor(Qt.yellow)
                        text.setFont(QFont("Arial", self.font_size))  # 글자 크기를 12로 설정
                        text.setPos((start_x + end_x)/2 + 5, (start_y + end_y)/2)

            # SiGe 레이어 두께 측정 표시
            for measurement in self.selectivity_data['pre_si_thickness_measurements']:
                if 'start_point' in measurement and 'end_point' in measurement:
                    start_x, start_y = measurement['start_point']
                    end_x, end_y = measurement['end_point']
                    
                    # 측정선 그리기
                    pen = QPen(Qt.yellow)
                    pen.setWidth(self.line_thickness)
                    self.scene.addLine(start_x, start_y, end_x, end_y, pen)
                    
                    # 두께 텍스트
                    if 'thickness_nm' in measurement:
                        text = self.scene.addText(f"{measurement['thickness_nm']:.2f}nm")
                        text.setDefaultTextColor(Qt.yellow)
                        text.setFont(QFont("Arial", self.font_size))
                        text.setPos((start_x + end_x)/2 + 5, (start_y + end_y)/2)

            # # Si/SiGe 센터 포인트 표시
            # for point_dict, color in [(self.selectivity_data['si_center_points'], Qt.red),
            #                         (self.selectivity_data['sige_center_points'], Qt.blue)]:
            #     pen = QPen(color)
            #     pen.setWidth(self.line_thickness)
            #     for point in point_dict.values():
            #         x, y = point
            #         self.scene.addEllipse(x, y, 6, 6, pen)

            # # Intersection 포인트 표시
            # for intersection in self.selectivity_data['si_intersections']:
            #     if 'intersection' in intersection:
            #         x, y = intersection['intersection']
            #         pen = QPen(Qt.magenta)
            #         pen.setWidth(self.line_thickness)
            #         self.scene.addEllipse(x, y, 4, 4, pen)

            # for intersection in self.selectivity_data['sige_intersections']:
            #     if 'intersection' in intersection:
            #         x, y = intersection['intersection']
            #         pen = QPen(Qt.cyan)
            #         pen.setWidth(self.line_thickness)
            #         self.scene.addEllipse(x, y, 4, 4, pen)

            # Best Intersection 결과 표시
            for sige_idx, result in self.selectivity_data['intersection_results'].items():
                if 'intersection' in result:
                    best_intersection = result['intersection']
                    # SiGe 센터 포인트 가져오기
                    sige_center = self.selectivity_data['sige_center_points'][sige_idx]
                    sige_recess = result['SiGe_Recess']
                    
                    # 교차점 표시
                    pen = QPen(Qt.magenta)
                    pen.setWidth(self.line_thickness)
                    self.scene.addEllipse(
                        best_intersection[0]-3, 
                        best_intersection[1]-3, 
                        6, 6, 
                        pen
                    )

                    # 거리 표시 라인
                    pen = QPen(Qt.yellow)
                    pen.setWidth(self.line_thickness)
                    self.scene.addLine(
                        sige_center[0],
                        sige_center[1],
                        best_intersection[0],
                        best_intersection[1],
                        pen
                    )

                    # 거리 텍스트
                    mid_x = (sige_center[0] + best_intersection[0]) / 2
                    mid_y = (sige_center[1] + best_intersection[1]) / 2
                    
                    # 그룹 아이템 생성
                    group = QGraphicsItemGroup()
                    self.scene.addItem(group)
                    
                    text = self.scene.addText(f"{sige_recess:.2f}nm")
                    text.setDefaultTextColor(Qt.yellow)
                    text.setFont(QFont("Arial", self.font_size))
                    
                    # 텍스트 배경 추가
                    text_rect = text.boundingRect()
                    background = self.scene.addRect(
                        text_rect,
                        QPen(Qt.NoPen),
                        QBrush(QColor(0, 0, 0, 128))
                    )
                    
                    # 텍스트와 배경을 그룹에 추가
                    group.addToGroup(background)
                    group.addToGroup(text)
                    
                    # 그룹 전체의 위치 설정
                    group.setPos(mid_x, mid_y)

        # 현재 작업 중인 포인트 표시
        if not self.show_all_layers:
            if len(self.points_add) > 0:
                pen = QPen(Qt.green)
                pen.setWidth(self.line_thickness)
                for point in self.points_add:
                    self.scene.addEllipse(point[0]-2, point[1]-2, 4, 4, pen)
                    
            if len(self.points_remove) > 0:
                pen = QPen(Qt.red)
                pen.setWidth(self.line_thickness)
                for point in self.points_remove:
                    self.scene.addEllipse(point[0]-2, point[1]-2, 4, 4, pen)
        
        # # 뷰 업데이트
        self.image_view.setSceneRect(self.scene.itemsBoundingRect())
        self.image_view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        self.update_layer_info()  # 정보 패널 업데이트

        # # 현재 레이어의 두께 측정 정보 표시
        # if self.current_layer is not None:
        #     current = self.layers[self.current_layer]
        #     if 'thickness_measurements' in current:
        #         for measurement in current['thickness_measurements']:
        #             start_point = measurement['start_point']
        #             end_point = measurement['end_point']
        #             distance = measurement['distance']
        #             tangent_points = measurement['tangent_points']
        #             slope = measurement['slope']
                    
        #             self.draw_thickness_measurement(
        #                 start_point[0], start_point[1],
        #                 end_point[0], end_point[1],
        #                 distance,
        #                 tangent_points,
        #                 slope
        #             )
        
        if hasattr(self, 'zoom_factor') and self.zoom_factor != 1.0:
            self.image_view.setTransform(QTransform().scale(self.zoom_factor, self.zoom_factor))

    def keyPressEvent(self, event):
        # 단순화된 Ctrl+Z 체크
        if (event.key() == Qt.Key_Z) and (event.modifiers() & Qt.ControlModifier):
            print("Undo triggered")  # 디버깅용
            self.undo()
            event.accept()
        else:
            super().keyPressEvent(event)

    def create_result_image(self):
            """마스크 결과 이미지 생성"""
            if self.image is None:
                return None
            
            # 설정값 가져오기
            line_thickness = getattr(self, 'line_thickness', 2)
            font_size = getattr(self, 'font_size', 30) / 25  # GUI의 font_size를 OpenCV 스케일로 변환
            
            # 원본 이미지 복사
            result = self.image.copy()
            height, width = self.image.shape[:2]
            
            # 모든 레이어의 마스크 합치기
            for layer_idx, layer in enumerate(self.layers):
                if layer['layer_type'] == 'Si':
                    if 'masks' in layer and len(layer['masks']) > 0:
                        mask = layer['masks'][0]
                        color = self.layer_colors[layer_idx % len(self.layer_colors)]
                        overlay = np.zeros_like(result)
                        overlay[mask] = color
                        result = cv2.addWeighted(result, 1, overlay, 0.1, 0)

            # Selectivity 계산 결과 표시
            if hasattr(self, 'selectivity_data'):
                # Si 레이어 두께 측정 표시
                for measurement in self.selectivity_data['post_si_thickness_measurements']:
                    if 'start_point' in measurement and 'end_point' in measurement:
                        start_point = tuple(map(int, measurement['start_point']))
                        end_point = tuple(map(int, measurement['end_point']))
                        
                        # 측정선 그리기
                        cv2.line(result, start_point, end_point, (0, 255, 255), line_thickness,cv2.LINE_4)
                        
                        # 두께 텍스트
                        if 'thickness_nm' in measurement:
                            mid_x = int((start_point[0] + end_point[0]) / 2)
                            mid_y = int((start_point[1] + end_point[1]) / 2)
                            text = f"{measurement['thickness_nm']:.2f}nm"

                            # 텍스트 크기 계산
                            (text_width, text_height), _ = cv2.getTextSize(
                                text, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
                            cv2.rectangle(result, 
                                        (mid_x + 15, mid_y - text_height - 10),
                                        (mid_x + text_width + 25, mid_y + 20),
                                        (0, 0, 0), -1)
                            cv2.putText(result, text, (mid_x + 30, mid_y + 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), line_thickness)

                # SiGe 레이어 두께 측정 표시
                for measurement in self.selectivity_data['pre_si_thickness_measurements']:
                    if 'start_point' in measurement and 'end_point' in measurement:
                        start_point = tuple(map(int, measurement['start_point']))
                        end_point = tuple(map(int, measurement['end_point']))
                        
                        # 측정선 그리기
                        cv2.line(result, start_point, end_point, (0, 255, 255), line_thickness,cv2.LINE_AA)
                        
                        # 두께 텍스트
                        if 'thickness_nm' in measurement:
                            mid_x = int((start_point[0] + end_point[0]) / 2)
                            mid_y = int((start_point[1] + end_point[1]) / 2)
                            text = f"{measurement['thickness_nm']:.2f}nm"

                            # 텍스트 크기 계산
                            (text_width, text_height), _ = cv2.getTextSize(
                                text, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
                            # 텍스트 배경을 위한 검은색 사각형
                            cv2.rectangle(result, 
                                        (mid_x + 15, mid_y - text_height - 10),
                                        (mid_x + text_width + 25, mid_y + 20),
                                        (0, 0, 0), -1)
                            cv2.putText(result, text, (mid_x + 30, mid_y + 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), line_thickness)

                # Best Intersection 결과 표시
                for sige_idx, result_data in self.selectivity_data['intersection_results'].items():
                    if 'intersection' in result_data:
                        best_intersection = tuple(map(int, result_data['intersection']))
                        sige_center = tuple(map(int, self.selectivity_data['sige_center_points'][sige_idx]))
                        si_center = tuple(map(int, self.selectivity_data['si_center_points'][result_data['si_layer']]))
                        sige_recess = result_data['SiGe_Recess']

                        # 교차점 표시
                        cv2.circle(result, best_intersection, line_thickness, (255, 0, 255), -1)
                        
                        # 거리 표시 라인
                        cv2.line(result, sige_center, best_intersection, (0, 255, 255), line_thickness, cv2.LINE_AA)
                        
                        # 거리 텍스트
                        mid_x = int((sige_center[0] + best_intersection[0]) / 2)
                        mid_y = int((sige_center[1] + best_intersection[1]) / 2)
                        text = f"{sige_recess:.2f}nm"
                        
                        # 텍스트 크기 계산
                        (text_width, text_height), _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, font_size, line_thickness)
                        
                        # 텍스트 배경을 위한 검은색 사각형
                        cv2.rectangle(result, 
                                    (mid_x - 5, mid_y - text_height - 5),
                                    (mid_x + text_width + 5, mid_y + 5),
                                    (0, 0, 0), -1)
                        
                        # 텍스트 그리기
                        cv2.putText(result, text, (mid_x, mid_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 255), line_thickness)

            return result
            
    def save_result(self):
        import time
        start_time = time.time()

        if self.image is None or not self.layers:
            QMessageBox.warning(self, "Warning", "저장할 결과가 없습니다.")
            return
        
        # 기본 파일 이름 설정
        setup_time = time.time()
        print(f"Initial setup took: {setup_time - start_time:.2f} seconds")

        default_name = "result"
        if hasattr(self, 'current_image_path'):
            default_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
            if default_name.lower().endswith('.pdf'):
                default_name = default_name[:-4]
        
        # 파일 저장 다이얼로그
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", 
            os.path.join(os.path.dirname(getattr(self, 'current_image_path', '')), default_name),
            "PDF Files (*.pdf);;Image Files (*.png);;All Files (*)")
        
        dialog_time = time.time()
        print(f"File dialog took: {dialog_time - setup_time:.2f} seconds")

        if not file_path:
            return
            
        try:
            base_path = os.path.splitext(file_path)[0]
            
            if file_path.lower().endswith('.pdf'):
                # PDF 리포트 생성 시간 측정
                pdf_start = time.time()
                self.generate_pdf_report(file_path)
                pdf_end = time.time()
                print(f"PDF generation took: {pdf_end - pdf_start:.2f} seconds")

                # NPZ 저장 시간 측정
                npz_start = time.time()
                self.save_layers_data(f"{base_path}.npz")
                npz_end = time.time()
                print(f"NPZ save took: {npz_end - npz_start:.2f} seconds")
            else:
                # 이미지 저장 시간 측정
                img_start = time.time()
                result_image = self.create_result_image()
                if result_image is not None:
                    cv2.imwrite(file_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                img_end = time.time()
                print(f"Image save took: {img_end - img_start:.2f} seconds")

                # NPZ 저장 시간 측정
                npz_start = time.time()
                if base_path.lower().endswith('.pdf'):
                    base_path = base_path[:-4]
                self.save_layers_data(f"{base_path}.npz")
                npz_end = time.time()
                print(f"NPZ save took: {npz_end - npz_start:.2f} seconds")
                
            # CSV 저장 시간 측정
            if hasattr(self, 'selectivity_data'):
                csv_start = time.time()
                self.save_csv_data(f"{base_path}.csv")
                csv_end = time.time()
                print(f"CSV save took: {csv_end - csv_start:.2f} seconds")
            
            end_time = time.time()
            print(f"Total save operation took: {end_time - start_time:.2f} seconds")
            
            QMessageBox.information(self, "Success", "결과가 성공적으로 저장되었습니다.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"저장 실패: {str(e)}")

    def save_csv_data(self, file_path):
        """Selectivity 데이터를 CSV 파일로 저장"""
        import pandas as pd
        
        if not hasattr(self, 'selectivity_data'):
            return
            
        # 데이터 추출
        sige_recess = []
        post_si = []
        pre_si = []

        for i in self.selectivity_data['intersection_results'].keys():
            try:
                sige_recess.append(self.selectivity_data['intersection_results'][i]['SiGe_Recess'])
            except:
                continue
                
        for i in range(len(self.selectivity_data['post_si_thickness_measurements'])):
            try:
                post_si.append(self.selectivity_data['post_si_thickness_measurements'][i]['thickness_nm'])
            except:
                continue

        for i in range(len(self.selectivity_data['pre_si_thickness_measurements'])):
            try:
                pre_si.append(self.selectivity_data['pre_si_thickness_measurements'][i]['thickness_nm'])
            except:
                continue

        df = pd.DataFrame({
            'SiGe_Recess (nm)': sige_recess,
            'Post_Si_Thickness (nm)': post_si, 
            'Pre_Si_Thickness (nm)': pre_si
        })

        df.index = range(1, 1 + len(df))
        df['Si Loss (nm)'] = df['Pre_Si_Thickness (nm)'] - df['Post_Si_Thickness (nm)']
        df.loc['Avg'] = df.mean()
        df.loc['Range(max - min)'] = df.max() - df.min()
        df.loc['Uniformity = (range/2*avg)*100'] = df.loc['Range(max - min)'] / (2 * df.loc['Avg']) * 100

        # CSV 파일로 저장
        df.to_csv(file_path)

    def save_layers_data(self, file_path=None):
        """현재 레이어 데이터를 파일로 저장"""
        if file_path is None:
            # 저장 경로 선택
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Layers Data", "", "Layers Data (*.npz);;All Files (*)")
            
            if not file_path:
                return
            
            # .npz 확장자 확인 및 추가
            if not file_path.lower().endswith('.npz'):
                file_path += '.npz'
        
        # 저장할 데이터 준비
        layers_data = []
        for layer in self.layers:
            layer_copy = layer.copy()
            
            # 마스크는 별도로 저장
            if 'masks' in layer_copy and layer_copy['masks']:
                layer_copy['has_masks'] = True
            else:
                layer_copy['has_masks'] = False
            
            layer_copy.pop('masks', None)
            layers_data.append(layer_copy)
        
        # 마스크 배열 준비
        masks = []
        for layer in self.layers:
            if 'masks' in layer and layer['masks']:
                masks.append(layer['masks'][0])
            else:
                masks.append(None)
        
        # 현재 설정 저장
        settings = {
            'pixel_to_nm': self.pixel_to_nm,
            'current_layer': self.current_layer,
            'mask_opacity': getattr(self, 'mask_opacity', 50),
            'line_thickness': getattr(self, 'line_thickness', 2),
            'font_size': getattr(self, 'font_size', 10),
        }
        
        # selectivity_data가 있다면 저장
        if hasattr(self, 'selectivity_data'):
            settings['selectivity_data'] = self.selectivity_data
        
        # 파일로 저장
        np.savez_compressed(
            file_path,
            layers=layers_data,
            masks=masks,
            settings=settings
        )


    def generate_pdf_report(self, file_path):
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from io import BytesIO
        import datetime

        # 현재 QGraphicsView의 크기 가져오기
        view_width = self.image_view.width()
        view_height = self.image_view.height()
        
        # A4 페이지 크기 계산 (포인트 단위)
        a4_width, a4_height = A4
        
        # 여백 설정
        margin = 36  # 1 inch = 72 points
        available_width = a4_width - 2 * margin
        available_height = a4_height - 2 * margin
        
        # 이미지 비율 계산
        image_ratio = float(self.image.shape[1]) / float(self.image.shape[0])
        
        # PDF 문서 생성
        doc = SimpleDocTemplate(
            file_path,
            pagesize=A4,
            rightMargin=margin,
            leftMargin=margin,
            topMargin=margin,
            bottomMargin=margin
        )
        
        elements = []
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        heading_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # 제목 추가
        elements.append(Paragraph("ThinFilm Analysis Report", title_style))
        elements.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        elements.append(Spacer(1, 20))
        
        try:
            # 결과 이미지 생성
            result_image = self.create_result_image()
            
            # BytesIO를 사용하여 메모리에서 이미지 처리
            img_buffer = BytesIO()
            
            # BGR에서 RGB로 변환
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # PIL Image로 변환
            from PIL import Image as PILImage
            pil_image = PILImage.fromarray(result_image)
            
            # 이미지 크기 계산
            # available_width를 기준으로 비율 유지하면서 크기 조정
            img_width = available_width
            img_height = img_width / image_ratio
            
            # 이미지가 너무 크면 available_height 기준으로 조정
            if img_height > available_height:
                img_height = available_height
                img_width = img_height * image_ratio
                
            # PIL Image 저장
            pil_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # reportlab Image 객체 생성
            img = Image(img_buffer)
            img.drawWidth = img_width
            img.drawHeight = img_height
            elements.append(img)
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            traceback.print_exc()

        # RGB 색상을 16진수 문자열로 변환하는 헬퍼 함수
        def rgb2hex(rgb):
            return f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'
        
        # 레이어 범례 추가
        legend_text = []
        for layer_idx, layer in enumerate(self.layers):
            if 'masks' in layer and len(layer['masks']) > 0 and layer['layer_type'] =='Si':
                # layer_colors에서 해당 레이어의 색상 가져오기
                color = self.layer_colors[layer_idx % len(self.layer_colors)]
                # RGB로 변환 (reportlab은 RGB 사용)
                color_rgb = (color[2]/255, color[1]/255, color[0]/255)  # BGR to RGB
                
                # 범례 텍스트 추가
                legend_text.append(
                    Paragraph(
                        f'<para><font color="{rgb2hex(color_rgb)}">■</font> Si Layer {layer_idx + 1}</para>',
                        styles["Normal"]
                    )
                )
        
        # 범례를 테이블로 구성
        if legend_text:
            elements.append(Spacer(1, 20))
            legend_data = []
            row = []
            for i, text in enumerate(legend_text):
                row.append(text)
                if (i + 1) % 2 == 0 or i == len(legend_text) - 1:
                    if len(row) < 2:
                        row.append('')  # 빈 셀 추가
                    legend_data.append(row)
                    row = []
            
            # 테이블 스타일 설정
            legend_table = Table(legend_data, colWidths=[250, 250])
            legend_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ]))
            elements.append(legend_table)

        # 범례 테이블 다음에 Selectivity 데이터 테이블 추가
        if hasattr(self, 'selectivity_data'):
            elements.append(Spacer(1, 20))
            elements.append(Paragraph("Selectivity Analysis Results", heading_style))
            elements.append(Spacer(1, 10))

            # 데이터 추출
            sige_recess = []
            post_si = []
            pre_si = []

            for i in self.selectivity_data['intersection_results'].keys():
                try:
                    sige_recess.append(self.selectivity_data['intersection_results'][i]['SiGe_Recess'])
                except:
                    continue
                    
            for i in range(len(self.selectivity_data['post_si_thickness_measurements'])):
                try:
                    post_si.append(self.selectivity_data['post_si_thickness_measurements'][i]['thickness_nm'])
                except:
                    continue

            for i in range(len(self.selectivity_data['pre_si_thickness_measurements'])):
                try:
                    pre_si.append(self.selectivity_data['pre_si_thickness_measurements'][i]['thickness_nm'])
                except:
                    continue

            # Si Loss 계산
            si_loss = [pre - post for pre, post in zip(pre_si, post_si)]

            # 테이블 데이터 생성
            table_data = [
                ['Layer', 'SiGe\nRecess (nm)', 'Pre Si\n(nm)', 'Post Si\n(nm)', 'Si Loss\n(nm)'],
            ]

            # 측정 데이터 추가
            for idx in range(len(sige_recess)):
                selectivity = sige_recess[idx]/si_loss[idx] if si_loss[idx] != 0 else 0
                row = [
                    f"{idx + 1}",
                    f"{sige_recess[idx]:.2f}",
                    f"{pre_si[idx]:.2f}",
                    f"{post_si[idx]:.2f}",
                    f"{si_loss[idx]:.2f}"
                ]
                table_data.append(row)

            # 통계 데이터 추가
            avg_sige = np.mean(sige_recess)
            avg_si_loss = np.mean(si_loss)
            
            range_sige = max(sige_recess) - min(sige_recess)
            range_si_loss = max(si_loss) - min(si_loss)
            
            unif_sige = (range_sige/(2*avg_sige))*100
            unif_si_loss = (range_si_loss/(2*avg_si_loss))*100

            selectivity = avg_sige / avg_si_loss

            table_data.extend([
                ['Avg', f"{avg_sige:.2f}", f"{np.mean(pre_si):.2f}", f"{np.mean(post_si):.2f}", f"{avg_si_loss:.2f}"],
                ['Range', f"{range_sige:.2f}", "-", "-", f"{range_si_loss:.2f}"],
                ['Unif.(%)', f"{unif_sige:.2f}", "-", "-", f"{unif_si_loss:.2f}"],
                ['Selectivity', f"{selectivity:.2f}", "", "", ""]
            ])

            # 테이블 스타일 설정
            col_widths = [60, 70, 70, 70, 70, 70]  # 각 열의 너비 조정
            selectivity_table = Table(table_data, colWidths=col_widths)
            
            # 기본 스타일
            table_style = [
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                
                # 헤더 스타일
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                
                # 통계 행 스타일
                ('SPAN', (1, -1), (4, -1)),
                ('BACKGROUND', (0, -4), (-1, -1), colors.lightgrey),
                ('FONTNAME', (0, -4), (-1, -1), 'Helvetica-Bold'),
            ]
            
            selectivity_table.setStyle(TableStyle(table_style))
            elements.append(selectivity_table)
        
        # elements.append(Spacer(1, 20))
        
        # # 레이어별 분석 정보
        # elements.append(Paragraph("Si Layer Analysis", heading_style))
        # elements.append(Spacer(1, 10))
        
        # table_style = TableStyle([
        #     ('GRID', (0, 0), (-1, -1), 1, colors.black),
        #     ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        #     ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        #     ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        #     ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        #     ('FONTSIZE', (0, 0), (-1, 0), 12),
        #     ('FONTSIZE', (0, 1), (-1, -1), 10),
        #     ('TOPPADDING', (0, 0), (-1, -1), 6),
        #     ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        #     ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        # ])

        # # 각 레이어별 분석
        # avg_thickness_dict = dict()
        # for i, layer in enumerate(self.layers):
        #     if layer['layer_type'] == 'Si':
        #         elements.append(Paragraph(f"Si Layer {i+1}", styles['Heading3']))
                
        #         if len(layer['masks']) > 0:
        #             # 평균 두께 분석
        #             avg_thickness = self.calculate_average_thickness(layer['masks'][0])
        #             if avg_thickness:
        #                 elements.append(Paragraph("Average Thickness Analysis", styles['Heading4']))
        #                 thickness_data = [
        #                     ["Parameter", "Value"],
        #                     ["Average Thickness", f"{avg_thickness['average']:.2f} nm"],
        #                     ["Standard Deviation", f"{avg_thickness['std']:.2f} nm"],
        #                     ["Valid Measurements", f"{avg_thickness['count']}"]
        #                 ]
        #                 if layer['layer_type'] =='Si':
        #                     avg_thickness_dict[i] = avg_thickness['thickness']
        #                 thickness_table = Table(thickness_data, style=table_style)
        #                 elements.append(thickness_table)
        #                 elements.append(Spacer(1, 20))

        #                 thickness_plot = Image(avg_thickness['plot'])
        #                 thickness_plot.drawWidth = 400
        #                 thickness_plot.drawHeight = 100
        #                 elements.append(thickness_plot)
        #                 elements.append(Spacer(1, 20))
        
        # fig = plt.figure(figsize=(10, 3))
        # for i in avg_thickness_dict.keys():
        #     plt.plot(pd.Series(avg_thickness_dict[i]).rolling(window=20).mean(), label = f'Layer {i}')

        # plt.legend(bbox_to_anchor=[1.05,1.05])
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        
        # # 이미지를 BytesIO에 저장
        # buf = BytesIO()
        # plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        # plt.close(fig)
        # buf.seek(0)
 
        # avg_thickness_plot = Image(buf)
        # avg_thickness_plot.drawWidth = 400
        # avg_thickness_plot.drawHeight = 100
        # elements.append(avg_thickness_plot)
        # # 푸터 추가
        # elements.append(Spacer(1, 20))
        # elements.append(Paragraph("Generated by KETI Segmentation Tool", normal_style))
        
        # PDF 생성
        try:
            doc.build(elements)
        except Exception as e:
            print(f"Error building PDF: {str(e)}")
            traceback.print_exc()
        finally:
            img_buffer.close()

    def set_mode(self, mode):
        print(f"Setting mode to: {mode}")  # 디버깅용
        self.mode = mode
        self.box_mode = (mode == 'box')
        self.thickness_mode = (mode == 'thickness')
        self.brush_mode = (mode == 'brush')
        
        # 버튼 상태 업데이트
        self.add_btn.setStyleSheet(
            "background-color: #2ecc71;" if mode == 'add' else "")
        self.remove_btn.setStyleSheet(
            "background-color: #e74c3c;" if mode == 'remove' else "")
        self.box_btn.setStyleSheet(
            "background-color: #f39c12;" if mode == 'box' else "")
        self.brush_btn.setStyleSheet(
            "background-color: #9b59b6;" if mode == 'brush' else "")
        self.thickness_btn.setStyleSheet(
            "background-color: #3498db;" if mode == 'thickness' else "")
        self.adjust_points_btn.setStyleSheet(
            "background-color: #3498db;" if mode == 'points' else "")

        # 브러시 설정 그룹 표시/숨김
        if hasattr(self, 'brush_settings_group'):
            self.brush_settings_group.setVisible(mode == 'brush')
        
        print(f"Current modes - thickness: {self.thickness_mode}, box: {self.box_mode}")  # 디버깅용


    def set_brush_size(self, size):
        """브러시 크기 설정"""
        self.brush_size = size

    def set_brush_type(self, checked):
        """브러시 타입 설정 (추가/제거)"""
        if checked:
            self.brush_add = True
        else:
            self.brush_add = False  

    def add_new_layer(self):

        # 이전 레이어의 타입을 확인
        previous_type = 'Si'  # 기본값
        if self.layers:  # 레이어가 존재하면
            previous_type = self.layers[-1].get('layer_type', 'Si')  # 마지막 레이어의 타입을 가져옴
        layer_num = len(self.layers) + 1
        new_layer = {
            'name': f'Layer {layer_num}',
            'masks': [],
            'points': [],
            'labels': [],
            'sizes': [],
            'points_add': [],
            'points_remove': [],
            'thickness_measurements': [],  # 두께 측정 저장
            'thickness_points': [],
            'thickness_values': [],
            'layer_type' : f'{previous_type}'
        }
        self.layers.append(new_layer)
        self.layer_list.addItem(new_layer['name'])
        self.current_layer = len(self.layers) - 1
        self.layer_list.setCurrentRow(self.current_layer)
            
    def delete_current_layer(self):
        current_row = self.layer_list.currentRow()
        if current_row >= 0:
            self.save_state()  # 상태 저장
            self.layers.pop(current_row)
            self.layer_list.takeItem(current_row)
            self.display_image()
                
    def change_layer(self, index):
        if 0 <= index < len(self.layers):
            self.current_layer = index
            
            # 레이어 타입 콤보박스 업데이트
            current_type = self.layers[index].get('layer_type', 'Si')
            combo_index = 0 if current_type == 'Si' else 1
            self.layer_type_combo.setCurrentIndex(combo_index)
            
            self.display_image()

    def mouse_move_event(self, event):
        if not self.box_mode or not self.box_start:
            return
            
        # 마우스 위치를 씬 좌표로 변환
        scene_pos = self.image_view.mapToScene(event.pos())
        self.box_end = (int(scene_pos.x()), int(scene_pos.y()))
        self.update_box_preview()

    def mouse_release_event(self, event):
        if not self.box_mode or not self.box_start:
            return
            
        # 마우스 위치를 씬 좌표로 변환
        scene_pos = self.image_view.mapToScene(event.pos())
        self.box_end = (int(scene_pos.x()), int(scene_pos.y()))
        
        if self.box_end != self.box_start:
            self.process_box()
        
        # 프리뷰 박스 제거
        if self.current_box:
            self.scene.removeItem(self.current_box)
            self.current_box = None
        
        self.box_start = None
        self.box_end = None

    def calculate_mask_size(self, mask):
        pixel_count = np.sum(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        percentage = (pixel_count / total_pixels) * 100
        return pixel_count, percentage        
        # 프리뷰 박스 제거
        if hasattr(self, 'current_box') and self.current_box is not None:
            try:
                self.scene.removeItem(self.current_box)
            except:
                pass
            self.current_box = None
        
        # 상태 초기화
        self.box_start = None
        self.box_end = None
            
    def add_point(self, x, y):
        if self.current_layer is None or self.current_layer >= len(self.layers):
            QMessageBox.warning(self, "Warning", "활성화된 레이어가 없습니다.")
            return
        
        self.save_state()  # 상태 저장
        current = self.layers[self.current_layer]
        if self.mode == 'add':
            current['points_add'].append([x, y])
        elif self.mode == 'remove':
            current['points_remove'].append([x, y])
        self.update_mask()
            
    def process_box(self):
        if not self.box_start or not self.box_end:
            return
        
        self.save_state()  # 상태 저장
        
        x1, y1 = self.box_start
        x2, y2 = self.box_end
        
        # 박스 좌표 계산
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        box = np.array([x_min, y_min, x_max, y_max])
        
        current = self.layers[self.current_layer]
        
        try:
            # SAM 예측 수행
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=True
            )
            
            # 최적의 마스크 선택
            if len(masks) > 0:
                best_mask_idx = scores.argmax()
                current['masks'] = [masks[best_mask_idx]]
                pixel_count, percentage = self.calculate_mask_size(masks[best_mask_idx])
                current['sizes'] = [(pixel_count, percentage)]
                
                # 박스 정보 저장
                current['box'] = box.tolist()
                
                # 화면 업데이트
                self.display_image()
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"마스크 생성 중 오류 발생: {str(e)}")

    def update_mask(self):
        if self.image is None or self.current_layer is None:
            return
                    
        current = self.layers[self.current_layer]
        
        if len(current['points_add']) + len(current['points_remove']) > 0:
            try:
                # 현재 레이어의 포인트만 사용
                combined_points = current['points_add'] + current['points_remove']
                combined_labels = [1] * len(current['points_add']) + [0] * len(current['points_remove'])
                
                masks, scores, logits = self.predictor.predict(
                    point_coords=np.array(combined_points),
                    point_labels=np.array(combined_labels),
                    multimask_output=self.multimask_output
                )
                
                if len(masks) > 0:
                    best_mask_idx = scores.argmax()
                    current['masks'] = [masks[best_mask_idx]]
                    pixel_count, percentage = self.calculate_mask_size(masks[best_mask_idx])
                    current['sizes'] = [(pixel_count, percentage)]
                    
                    # 포인트 정보 저장
                    current['points'] = combined_points
                    current['labels'] = combined_labels
                
                self.display_image()
                    
            except Exception as e:
                print(f"마스크 생성 중 오류 발생: {str(e)}")

    def show_scale_dialog(self):
        dialog = ScaleSettingsDialog(self)
        dialog.exec_()

    def show_analysis_dialog(self):
        dialog = AnalysisParametersDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # 분석 파라미터 업데이트
            self.confidence_threshold = dialog.confidence_threshold.value()
            self.multimask_output = dialog.multimask_output.isChecked()
            self.window_size = dialog.window_size.value()
            # 필요한 경우 관련 함수 업데이트

    def show_appearance_dialog(self):
        dialog = AppearanceSettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # 외관 설정 업데이트
            self.mask_opacity = dialog.mask_opacity.value()
            self.line_thickness = dialog.line_thickness.value()
            self.font_size = dialog.font_size.value()
            self.display_image()  # 화면 업데이트

    def show_report_dialog(self):
        dialog = ReportSettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.is_include_roughness = dialog.is_include_roughness.value()

    def measure_thickness_at_intersection(self, mask, x, y):
        """교차점에서 두께 측정 (measure_thickness_at_x와 유사)"""
        try:
            if not mask.any():
                return None
                
            mask_uint8 = mask.astype(np.uint8) * 255
            height, width = mask_uint8.shape
            
            if x < 0 or x >= width or y < 0 or y >= height:
                return None
                
            # 해당 x 좌표에서 마스크 내 y 좌표 찾기
            y_coords = np.where(mask_uint8[:, int(x)] > 0)[0]
            if len(y_coords) == 0:
                return None
                
            # 마스크 내에서 가장 아래 점 찾기
            y_bottom = np.max(y_coords)
            
            # 접선 계산을 위한 설정
            window_size = 150
            nearby_points_x = []
            nearby_points_y = []
            
            # 주변 점들 수집 (접선 계산용)
            for dx in range(-window_size, window_size+1):
                curr_x = int(x) + dx
                if 0 <= curr_x < width:
                    curr_y_coords = np.where(mask_uint8[:, curr_x] > 0)[0]
                    if len(curr_y_coords) > 0:
                        curr_y = np.max(curr_y_coords)  # 하단 경계
                        nearby_points_x.append(curr_x)
                        nearby_points_y.append(curr_y)
            
            if len(nearby_points_x) > 2:
                # 선형 회귀로 접선의 기울기 계산
                coeffs = np.polyfit(nearby_points_x, nearby_points_y, 1)
                slope = coeffs[0]
                intercept = coeffs[1]
                
                # 접선 그리기용 점 계산
                x1 = int(x) - window_size
                x2 = int(x) + window_size
                y1 = int(slope * x1 + intercept)
                y2 = int(slope * x2 + intercept)
                
                # 수직선 관련 계산
                if abs(slope) > 1e-6:  # 기울기가 0이 아닌 경우
                    perpendicular_slope = -1/slope
                    perpendicular_length = height
                    
                    # 교차점 찾기
                    intersection_found = False
                    last_valid_point = None
                    first_exit_point = None
                    
                    for t in range(0, abs(perpendicular_length)):
                        if perpendicular_slope > 0:
                            check_x = int(x - t/np.sqrt(1 + perpendicular_slope**2))
                            check_y = int(y_bottom - perpendicular_slope * t/np.sqrt(1 + perpendicular_slope**2))
                        else:
                            check_x = int(x + t/np.sqrt(1 + perpendicular_slope**2))
                            check_y = int(y_bottom + perpendicular_slope * t/np.sqrt(1 + perpendicular_slope**2))
                        
                        # 이미지 범위 체크
                        if (check_x < 0 or check_x >= width or 
                            check_y < 0 or check_y >= height):
                            break
                        
                        # 마스크 내부의 마지막 점 저장
                        if mask_uint8[check_y, check_x] > 0:
                            last_valid_point = (check_x, check_y)
                        # 마스크를 벗어난 첫 점 저장
                        elif last_valid_point is not None and first_exit_point is None:
                            first_exit_point = (check_x, check_y)
                            
                            # 윗면 근처인지 확인
                            y_vals = np.where(mask_uint8[:, check_x] > 0)[0]
                            if len(y_vals) > 0:
                                y_top = np.min(y_vals)
                                if abs(check_y - y_top) < 10:  # 윗면과 가까운 경우
                                    intersection_found = True
                                    distance = np.sqrt((check_x - x)**2 + (y_top - y_bottom)**2)
                                    distance_nm = distance * self.pixel_to_nm
                                    
                                    return {
                                        'thickness': distance,
                                        'thickness_nm': distance_nm,
                                        'start_point': [x, y_bottom],
                                        'end_point': [check_x, y_top],
                                        'tangent_points': [(x1, y1), (x2, y2)],
                                        'slope': slope
                                    }
                    
                    if not intersection_found and last_valid_point is not None:
                        distance = np.sqrt((last_valid_point[0] - x)**2 + 
                                        (last_valid_point[1] - y_bottom)**2)
                        distance_nm = distance * self.pixel_to_nm
                        
                        return {
                            'thickness': distance,
                            'thickness_nm': distance_nm,
                            'start_point': [x, y_bottom],
                            'end_point': [last_valid_point[0], last_valid_point[1]],
                            'tangent_points': [(x1, y1), (x2, y2)],
                            'slope': slope
                        }
            
            # 법선을 사용한 측정이 실패한 경우, 수직 방향 두께 측정
            y_top = np.min(y_coords)  # 상단 경계
            vertical_thickness = y_bottom - y_top
            
            return {
                'thickness': vertical_thickness,
                'thickness_nm': vertical_thickness * self.pixel_to_nm,
                'start_point': [x, y_bottom],
                'end_point': [x, y_top],
                'is_vertical': True
            }
            
        except Exception as e:
            print(f"Error measuring thickness at intersection: {str(e)}")
            traceback.print_exc()
            return None

    def determine_structure_position(self):
        """다층 구조가 좌측 또는 우측에 정렬되어 있는지 판단"""
        try:
            # 각 레이어의 좌우 여백 계산
            left_margins = []
            right_margins = []
            
            for layer in self.layers:
                if 'masks' in layer and len(layer['masks']) > 0:
                    mask = layer['masks'][0]
                    height, width = mask.shape
                    
                    # x축 방향으로 마스크가 있는 영역 찾기
                    x_indices = np.where(np.any(mask, axis=0))[0]
                    if len(x_indices) == 0:
                        continue
                        
                    # 좌우 여백 계산
                    left_margin = x_indices[0]  # 좌측 여백
                    right_margin = width - x_indices[-1] - 1  # 우측 여백
                    
                    left_margins.append(left_margin)
                    right_margins.append(right_margin)
            
            # 여백이 없으면 기본값 사용
            if not left_margins or not right_margins:
                return 'right'  # 기본값
                
            # 평균 여백으로 위치 판단
            avg_left = np.mean(left_margins)
            avg_right = np.mean(right_margins)
            
            # 좌측 여백이 더 작으면 '좌측 정렬', 아니면 '우측 정렬'
            return 'left' if avg_left < avg_right else 'right'
            
        except Exception as e:
            print(f"Error determining structure position: {str(e)}")
            traceback.print_exc()
            return 'right'  # 오류 발생 시 기본값

    def brush_mode_edit(self):
        """브러시 모드에서 레이어 편집 다이얼로그를 엽니다"""
        try:
            if self.current_layer is None or self.current_layer >= len(self.layers):
                QMessageBox.warning(self, "Warning", "활성화된 레이어가 없습니다.")
                return
                
            # 현재 레이어 정보
            current_layer = self.layers[self.current_layer]
            
            # 현재 마스크가 없으면 빈 마스크 생성
            if 'masks' not in current_layer or len(current_layer['masks']) == 0:
                height, width = self.image.shape[:2]
                current_layer['masks'] = [np.zeros((height, width), dtype=bool)]
                current_layer['sizes'] = [(0, 0.0)]
            
            # 상태 저장
            self.save_state()
            
            # 브러시 편집 다이얼로그 생성
            brush_dialog = QDialog(self)
            brush_dialog.setWindowTitle(f"Brush Edit - {current_layer['name']}")
            brush_dialog.setMinimumSize(1200, 800)
            
            layout = QVBoxLayout()
            
            # 설명 레이블 추가
            instruction_label = QLabel("브러시로 영역을 그리거나 지우세요. 브러시 크기는 슬라이더로 조절할 수 있습니다.")
            instruction_label.setStyleSheet("font-weight: bold; color: blue;")
            layout.addWidget(instruction_label)
            
            # 상단 컨트롤 패널
            control_panel = QWidget()
            control_layout = QHBoxLayout(control_panel)
            
            # 브러시 타입 선택 (추가/제거)
            brush_type_group = QGroupBox("Brush Type")
            brush_type_layout = QHBoxLayout()
            
            brush_add_radio = QRadioButton("Add")
            brush_remove_radio = QRadioButton("Remove")
            brush_add_radio.setChecked(True)
            
            brush_type_layout.addWidget(brush_add_radio)
            brush_type_layout.addWidget(brush_remove_radio)
            brush_type_group.setLayout(brush_type_layout)
            control_layout.addWidget(brush_type_group)
            
            # 브러시 크기 설정
            brush_size_group = QGroupBox("Brush Size")
            brush_size_layout = QHBoxLayout()
            
            brush_size_label = QLabel("Size:")
            brush_size_slider = QSlider(Qt.Horizontal)
            brush_size_slider.setRange(1, 100)
            brush_size_slider.setValue(self.brush_size)
            brush_size_value = QLabel(f"{self.brush_size}")
            
            # 슬라이더 값 변경 시 라벨 업데이트
            brush_size_slider.valueChanged.connect(
                lambda v: brush_size_value.setText(f"{v}"))
            
            brush_size_layout.addWidget(brush_size_label)
            brush_size_layout.addWidget(brush_size_slider)
            brush_size_layout.addWidget(brush_size_value)
            brush_size_group.setLayout(brush_size_layout)
            control_layout.addWidget(brush_size_group)
            
            # 레이어 정보 표시
            layer_info_group = QGroupBox("Layer Info")
            layer_info_layout = QVBoxLayout()
            
            layer_type = current_layer.get('layer_type', 'Si')
            layer_info_label = QLabel(f"Type: {layer_type}")
            if 'sizes' in current_layer and len(current_layer['sizes']) > 0:
                pixel_count, percentage = current_layer['sizes'][0]
                coverage_label = QLabel(f"Coverage: {percentage:.2f}%")
                layer_info_layout.addWidget(coverage_label)
            
            layer_info_layout.addWidget(layer_info_label)
            layer_info_group.setLayout(layer_info_layout)
            control_layout.addWidget(layer_info_group)
            
            # 도움말 버튼
            help_btn = QPushButton("Help")
            help_btn.clicked.connect(self.show_brush_help)
            control_layout.addWidget(help_btn)
            
            layout.addWidget(control_panel)
            
            # 사용자 정의 GraphicsView 클래스
            class BrushEditView(QGraphicsView):
                def __init__(self, parent_dialog, parent_app, brush_add_radio, brush_size_slider):
                    super().__init__()
                    self.parent_dialog = parent_dialog
                    self.parent_app = parent_app
                    self.brush_add_radio = brush_add_radio
                    self.brush_size_slider = brush_size_slider
                    self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
                    self.setDragMode(QGraphicsView.NoDrag)
                    self.setMouseTracking(True)
                    self.brush_active = False
                    self.last_pos = None
                    self.current_brush_preview = None
                    
                def mousePressEvent(self, event):
                    if event.button() == Qt.LeftButton:
                        self.brush_active = True
                        pos = self.mapToScene(event.pos())
                        x, y = int(pos.x()), int(pos.y())
                        
                        # 이미지 범위 확인
                        if 0 <= x < self.parent_app.image.shape[1] and 0 <= y < self.parent_app.image.shape[0]:
                            self.last_pos = (x, y)
                            # 브러시 적용
                            brush_add = self.brush_add_radio.isChecked()
                            brush_size = self.brush_size_slider.value()
                            self.parent_app.dialog_apply_brush(x, y, brush_size, brush_add)
                            print(f"Brush applied at ({x}, {y}), size: {brush_size}, add: {brush_add}")
                    super().mousePressEvent(event)
                    
                def mouseMoveEvent(self, event):
                    pos = self.mapToScene(event.pos())
                    x, y = int(pos.x()), int(pos.y())
                    
                    # 브러시 미리보기 업데이트
                    self.update_brush_preview(x, y)
                    
                    # 브러시가 활성화된 경우 그리기
                    if self.brush_active and self.last_pos:
                        # 이미지 범위 확인
                        if 0 <= x < self.parent_app.image.shape[1] and 0 <= y < self.parent_app.image.shape[0]:
                            prev_x, prev_y = self.last_pos
                            dx, dy = x - prev_x, y - prev_y
                            
                            # 두 점 사이의 거리
                            distance = max(1, int(((dx ** 2) + (dy ** 2)) ** 0.5))
                            
                            # 두 점 사이를 보간하여 브러시 적용
                            step_size = max(1, distance // 10)
                            brush_add = self.brush_add_radio.isChecked()
                            brush_size = self.brush_size_slider.value()
                            
                            for i in range(0, distance + 1, step_size):
                                ratio = i / distance if distance > 0 else 0
                                ix = int(prev_x + dx * ratio)
                                iy = int(prev_y + dy * ratio)
                                self.parent_app.dialog_apply_brush(ix, iy, brush_size, brush_add)
                            
                            self.last_pos = (x, y)
                    
                    super().mouseMoveEvent(event)
                    
                def mouseReleaseEvent(self, event):
                    if event.button() == Qt.LeftButton:
                        self.brush_active = False
                        self.last_pos = None
                    super().mouseReleaseEvent(event)
                    
                def update_brush_preview(self, x, y):
                    # 이전 브러시 미리보기 제거
                    if self.current_brush_preview:
                        self.scene().removeItem(self.current_brush_preview)
                        self.current_brush_preview = None
                    
                    # 새 브러시 미리보기 추가
                    brush_size = self.brush_size_slider.value()
                    brush_add = self.brush_add_radio.isChecked()
                    
                    pen = QPen(Qt.yellow if brush_add else Qt.red)
                    pen.setWidth(2)
                    pen.setStyle(Qt.DashLine)
                    
                    self.current_brush_preview = self.scene().addEllipse(
                        x - brush_size, y - brush_size,
                        brush_size * 2, brush_size * 2,
                        pen, QBrush(Qt.transparent)
                    )
                
                def keyPressEvent(self, event):
                    # 브러시 크기 단축키
                    if event.key() == Qt.Key_BracketLeft:  # [
                        current = self.brush_size_slider.value()
                        new_value = max(1, current - 5)
                        self.brush_size_slider.setValue(new_value)
                        event.accept()
                    elif event.key() == Qt.Key_BracketRight:  # ]
                        current = self.brush_size_slider.value()
                        new_value = min(100, current + 5)
                        self.brush_size_slider.setValue(new_value)
                        event.accept()
                    else:
                        super().keyPressEvent(event)
                        
                def wheelEvent(self, event):
                    if event.modifiers() == Qt.ControlModifier:
                        # Ctrl + 휠로 확대/축소
                        factor = 1.2
                        if event.angleDelta().y() < 0:
                            factor = 1.0 / factor
                            
                        self.scale(factor, factor)
                    else:
                        super().wheelEvent(event)
            
            # 그래픽 뷰와 씬 설정
            self.dialog_scene = QGraphicsScene()
            self.dialog_view = BrushEditView(brush_dialog, self, brush_add_radio, brush_size_slider)
            self.dialog_view.setScene(self.dialog_scene)
            
            # 원본 이미지 표시
            height, width = self.image.shape[:2]
            bytes_per_line = 3 * width
            q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.dialog_image_item = self.dialog_scene.addPixmap(pixmap)
            
            # 현재 마스크 표시
            current_mask = current_layer['masks'][0]
            mask_image = np.zeros((height, width, 4), dtype=np.uint8)
            layer_type = current_layer.get('layer_type', 'Si')
            
            # 레이어 타입에 따라 색상 설정
            if layer_type == 'Si':
                mask_color = [255, 0, 0, 128]  # 빨간색 (반투명)
            else:  # SiGe
                mask_color = [0, 0, 255, 128]  # 파란색 (반투명)
                
            mask_image[current_mask] = mask_color
            
            mask_qimg = QImage(mask_image.data, width, height, QImage.Format_RGBA8888)
            mask_pixmap = QPixmap.fromImage(mask_qimg)
            self.dialog_mask_item = self.dialog_scene.addPixmap(mask_pixmap)
            
            # 그래픽 뷰를 레이아웃에 추가
            layout.addWidget(self.dialog_view, stretch=1)
            
            # 버튼 영역
            button_layout = QHBoxLayout()
            
            # 실행 취소 버튼
            undo_btn = QPushButton("Undo")
            undo_btn.clicked.connect(self.dialog_undo)
            button_layout.addWidget(undo_btn)
            
            # 초기화 버튼
            reset_btn = QPushButton("Reset")
            reset_btn.clicked.connect(lambda: self.dialog_reset_mask(current_layer))
            button_layout.addWidget(reset_btn)
            
            button_layout.addStretch()
            
            # 확인 및 취소 버튼
            ok_btn = QPushButton("OK")
            ok_btn.clicked.connect(brush_dialog.accept)
            button_layout.addWidget(ok_btn)
            
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(brush_dialog.reject)
            button_layout.addWidget(cancel_btn)
            
            layout.addLayout(button_layout)
            
            brush_dialog.setLayout(layout)

            # 객체 참조 저장 - 이 부분이 중요합니다!
            self.temp_brush_add_radio = brush_add_radio
            self.temp_brush_remove_radio = brush_remove_radio
            self.temp_brush_size_slider = brush_size_slider
            
            # 다이얼로그 결과 처리
            result = brush_dialog.exec_()
            if result == QDialog.Rejected:
                # 취소 시 상태 복원
                self.undo()
            
            # 화면 새로고침
            self.display_image()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"브러시 편집 중 오류 발생: {str(e)}")
            traceback.print_exc()
            
    def dialog_apply_brush(self, x, y, brush_size, add_mode):
        """다이얼로그 내에서 브러시 적용"""
        try:
            if self.current_layer is None or self.current_layer >= len(self.layers):
                print("Invalid layer")
                return
                
            current = self.layers[self.current_layer]
            
            # 현재 마스크 가져오기
            if 'masks' not in current or len(current['masks']) == 0:
                print("No mask in layer")
                return
                
            mask = current['masks'][0].copy()
            
            # 브러시 영역 계산
            y_coords, x_coords = np.ogrid[-brush_size:brush_size+1, -brush_size:brush_size+1]
            brush_mask = x_coords**2 + y_coords**2 <= brush_size**2
            
            # 브러시 적용 범위 계산
            y_min = max(0, y - brush_size)
            y_max = min(mask.shape[0], y + brush_size + 1)
            x_min = max(0, x - brush_size)
            x_max = min(mask.shape[1], x + brush_size + 1)
            
            # 브러시 마스크 부분 계산
            brush_y_min = max(0, brush_size - y)
            brush_y_max = brush_y_min + (y_max - y_min)
            brush_x_min = max(0, brush_size - x)
            brush_x_max = brush_x_min + (x_max - x_min)
            
            # 경계 검사
            if (brush_y_min < 0 or brush_y_max > brush_mask.shape[0] or 
                brush_x_min < 0 or brush_x_max > brush_mask.shape[1]):
                print(f"Brush boundary error: {brush_y_min}, {brush_y_max}, {brush_x_min}, {brush_x_max}")
                return
            
            # 브러시 적용 부분
            brush_part = brush_mask[brush_y_min:brush_y_max, brush_x_min:brush_x_max]
            
            # 적용할 영역의 크기 확인
            if brush_part.shape[0] != y_max - y_min or brush_part.shape[1] != x_max - x_min:
                print(f"Shape mismatch: brush_part {brush_part.shape}, target area: {y_max-y_min}x{x_max-x_min}")
                return
            
            # 마스크 수정 (추가 또는 제거)
            if add_mode:
                mask[y_min:y_max, x_min:x_max][brush_part] = True
            else:
                mask[y_min:y_max, x_min:x_max][brush_part] = False
            
            # 마스크 업데이트
            current['masks'][0] = mask
            
            # 픽셀 수와 퍼센트 계산
            pixel_count, percentage = self.calculate_mask_size(mask)
            current['sizes'][0] = (pixel_count, percentage)
            
            # 다이얼로그 내의 마스크 표시 업데이트
            self.update_dialog_mask()
            
            print(f"Brush applied at ({x}, {y}), size: {brush_size}, add: {add_mode}, " +
                f"pixel count: {pixel_count}, percentage: {percentage:.2f}%")
        
        except Exception as e:
            print(f"Error in dialog_apply_brush: {str(e)}")
            traceback.print_exc()
                
    def update_dialog_mask(self):
        """다이얼로그 내의 마스크 표시 업데이트"""
        try:
            if not hasattr(self, 'dialog_mask_item') or not hasattr(self, 'current_layer'):
                print("Dialog mask item or current layer not found")
                return
                
            current = self.layers[self.current_layer]
            if 'masks' not in current or len(current['masks']) == 0:
                print("No mask in current layer")
                return
                
            current_mask = current['masks'][0]
            height, width = self.image.shape[:2]
            mask_image = np.zeros((height, width, 4), dtype=np.uint8)
            
            layer_type = current.get('layer_type', 'Si')
            if layer_type == 'Si':
                mask_color = [255, 0, 0, 128]  # 빨간색 (반투명)
            else:  # SiGe
                mask_color = [0, 0, 255, 128]  # 파란색 (반투명)
            
            # 마스크 적용
            mask_image[current_mask] = mask_color
            
            # QImage 생성 및 pixmap 업데이트
            mask_qimg = QImage(mask_image.data, width, height, QImage.Format_RGBA8888)
            mask_pixmap = QPixmap.fromImage(mask_qimg)
            
            # pixmap 업데이트
            self.dialog_mask_item.setPixmap(mask_pixmap)
            
            # 다이얼로그 씬 업데이트
            if hasattr(self, 'dialog_scene'):
                self.dialog_scene.update()
                
            print(f"Mask updated: {np.sum(current_mask)} pixels, {100 * np.sum(current_mask) / (height * width):.2f}% coverage")
        
        except Exception as e:
            print(f"Error in update_dialog_mask: {str(e)}")
            traceback.print_exc()
            
    def dialog_undo(self):
        """다이얼로그 내에서 실행 취소"""
        if len(self.history) > 0:
            previous_state = self.history.pop()
            
            self.layers = [layer.copy() for layer in previous_state['layers']]
            self.current_layer = previous_state['current_layer']
            
            # 다이얼로그 마스크 업데이트
            self.update_dialog_mask()

    def dialog_reset_mask(self, layer):
        """다이얼로그 내에서 마스크 초기화"""
        # 상태 저장
        self.save_state()
        
        # 마스크 초기화
        height, width = self.image.shape[:2]
        layer['masks'][0] = np.zeros((height, width), dtype=bool)
        layer['sizes'][0] = (0, 0.0)
        
        # 다이얼로그 마스크 업데이트
        self.update_dialog_mask()

    def show_brush_help(self):
        """브러시 도움말 표시"""
        help_text = """
        <h3>브러시 툴 사용법</h3>
        <p><b>기본 사용법:</b> 마우스 드래그로 그리기/지우기</p>
        <p><b>브러시 타입:</b> Add(추가) 또는 Remove(제거) 선택</p>
        <p><b>브러시 크기:</b> 슬라이더로 조절 가능</p>
        <p><b>단축키:</b></p>
        <ul>
            <li>[: 브러시 크기 줄이기</li>
            <li>]: 브러시 크기 늘리기</li>
            <li>Ctrl + 마우스 휠: 확대/축소</li>
        </ul>
        """
        
        QMessageBox.information(self, "Brush Tool Help", help_text)
        
    def adjust_layer_points(self):
        """Si 레이어 경계점과 SiGe 센터 지점을 표시하고 조정"""
        try:
            # 데이터 준비
            layers_data = []
            masks = []
            
            for i, layer in enumerate(self.layers):
                if 'masks' in layer and len(layer['masks']) > 0:
                    layer_info = {
                        'name': layer['name'],
                        'layer_type': layer.get('layer_type', 'Si'),
                    }
                    layers_data.append(layer_info)
                    masks.append(layer['masks'][0])
                
            if len(layers_data) < 2:
                QMessageBox.warning(self, "Warning", "최소 2개 이상의 레이어가 필요합니다.")
                return
                
            # Si와 SiGe 레이어 분류
            si_layer = [i for i, layer_data in enumerate(layers_data) if layer_data['layer_type'] == 'Si']
            sige_layer = [i for i, layer_data in enumerate(layers_data) if layer_data['layer_type'] == 'SiGe']
            
            if not si_layer or not sige_layer:
                QMessageBox.warning(self, "Warning", "Si와 SiGe 타입의 레이어가 각각 필요합니다.")
                return
                
            # 구조 방향 확인
            is_right_structure = self.determine_structure_position() == 'right'
            
            # Si 레이어 경계점 계산
            si_endpoints = {}
            for i in si_layer:
                mask = masks[i]
                y_coords, x_coords = np.where(mask > 0)
                
                if len(x_coords) == 0:
                    continue
                    
                left_x = np.min(x_coords)
                left_y = np.mean(y_coords[x_coords == left_x])
                right_x = np.max(x_coords)
                right_y = np.mean(y_coords[x_coords == right_x])
                
                # 구조에 따라 중요한 경계점 선택
                if is_right_structure:
                    boundary_x = left_x  # 오른쪽 구조는 좌측 경계가 중요
                    boundary_y = left_y
                else:
                    boundary_x = right_x  # 왼쪽 구조는 우측 경계가 중요
                    boundary_y = right_y
                    
                si_endpoints[i] = {
                    'left': [left_x, left_y],
                    'right': [right_x, right_y],
                    'boundary': [boundary_x, boundary_y]
                }
            
            # SiGe 센터 계산
            sige_centers = {}
            for i in sige_layer:
                mask = masks[i]
                y_coords, x_coords = np.where(mask > 0)
                
                if len(x_coords) == 0:
                    continue
                    
                center_y = int(np.median(y_coords))
                x_values_at_center_y = x_coords[y_coords == center_y]
                
                # 오른쪽/왼쪽 구조에 따라 센터 선택
                if is_right_structure:
                    center_x = np.min(x_values_at_center_y)  # 오른쪽 구조
                else:
                    center_x = np.max(x_values_at_center_y)  # 왼쪽 구조
                    
                sige_centers[i] = [center_x, center_y]
            
            # 저장된 조정된 포인트 불러오기
            adjusted_si_endpoints = getattr(self, 'adjusted_si_endpoints', {})
            adjusted_sige_centers = getattr(self, 'adjusted_sige_centers', {})
            
            # 이전에 조정한 값을 기본값으로 사용
            for i, endpoint in si_endpoints.items():
                if i in adjusted_si_endpoints:
                    si_endpoints[i]['boundary'] = adjusted_si_endpoints[i]
                    
            for i, center in sige_centers.items():
                if i in adjusted_sige_centers:
                    sige_centers[i] = adjusted_sige_centers[i]
            
            # 조정 다이얼로그 생성
            adjust_dialog = QDialog(self)
            adjust_dialog.setWindowTitle("Adjust Layer Points")
            adjust_dialog.setMinimumSize(900, 700)
            
            layout = QVBoxLayout()
            
            # 설명 레이블 추가
            instruction_label = QLabel("점을 직접 드래그하여 위치를 조정하거나 스피너를 사용하세요.")
            instruction_label.setStyleSheet("font-weight: bold; color: blue;")
            layout.addWidget(instruction_label)
            
            # 사용자 정의 GraphicsView 클래스
            class DraggablePointsView(QGraphicsView):
                def __init__(self, parent=None):
                    super().__init__(parent)
                    self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
                    self.setDragMode(QGraphicsView.NoDrag)
                    self.setMouseTracking(True)
                    self.currentDragItem = None
                    self.lastMousePos = None
                    
                def mousePressEvent(self, event):
                    if event.button() == Qt.LeftButton:
                        pos = self.mapToScene(event.pos())
                        items = self.scene().items(pos)
                        for item in items:
                            if isinstance(item, QGraphicsEllipseItem):
                                self.currentDragItem = item
                                self.lastMousePos = pos
                                self.setCursor(Qt.ClosedHandCursor)
                                return
                    super().mousePressEvent(event)
                    
                def mouseMoveEvent(self, event):
                    if self.currentDragItem and self.lastMousePos:
                        newPos = self.mapToScene(event.pos())
                        delta = newPos - self.lastMousePos
                        
                        # 현재 위치 가져오기
                        rect = self.currentDragItem.rect()
                        center_x = rect.x() + rect.width()/2
                        center_y = rect.y() + rect.height()/2
                        
                        # 새 위치 계산
                        new_x = center_x + delta.x()
                        new_y = center_y + delta.y()
                        
                        # 점 이동
                        self.currentDragItem.setRect(new_x - rect.width()/2, 
                                                new_y - rect.height()/2, 
                                                rect.width(), rect.height())
                        
                        # 연결된 텍스트 이동
                        for item in self.scene().items():
                            if isinstance(item, QGraphicsTextItem) and item.data(0) == self.currentDragItem.data(0):
                                item.setPos(new_x + 10, new_y - 10)
                        
                        # 스피너 값 업데이트
                        data = self.currentDragItem.data(0)
                        if data:
                            layer_type, layer_idx = data.split('_')
                            layer_idx = int(layer_idx)
                            
                            if layer_type == 'si':
                                # Si 레이어 스피너 업데이트
                                si_spinners[layer_idx]['x'].setValue(int(new_x))
                                si_spinners[layer_idx]['y'].setValue(int(new_y))
                                # Si 데이터 업데이트
                                si_endpoints[layer_idx]['boundary'] = [new_x, new_y]
                            elif layer_type == 'sige':
                                # SiGe 레이어 스피너 업데이트
                                sige_spinners[layer_idx]['x'].setValue(int(new_x))
                                sige_spinners[layer_idx]['y'].setValue(int(new_y))
                                # SiGe 데이터 업데이트
                                sige_centers[layer_idx] = [new_x, new_y]
                        
                        self.lastMousePos = newPos
                        return
                        
                    # 점 위에 마우스가 있을 때 커서 변경
                    pos = self.mapToScene(event.pos())
                    items = self.scene().items(pos)
                    for item in items:
                        if isinstance(item, QGraphicsEllipseItem):
                            self.setCursor(Qt.OpenHandCursor)
                            return
                    self.setCursor(Qt.ArrowCursor)
                    
                    super().mouseMoveEvent(event)
                    
                def mouseReleaseEvent(self, event):
                    if event.button() == Qt.LeftButton and self.currentDragItem:
                        self.currentDragItem = None
                        self.lastMousePos = None
                        self.setCursor(Qt.ArrowCursor)
                    super().mouseReleaseEvent(event)
                    
            # 이미지 표시용 그래픽스 뷰
            graphics_view = DraggablePointsView()
            scene = QGraphicsScene()
            graphics_view.setScene(scene)
            layout.addWidget(graphics_view, stretch=3)
            
            # 이미지 및 포인트 표시
            image = self.create_result_image()  # 결과 이미지 생성
            height, width = image.shape[:2]
            
            # QImage로 변환
            image_qimg = QImage(image.data, width, height, width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image_qimg)
            scene.addPixmap(pixmap)
            
            # 경계점 표시용 아이템들
            si_point_items = {}
            sige_point_items = {}
            
            # 스피너 저장용 딕셔너리
            si_spinners = {}
            sige_spinners = {}
            
            # Si 경계점 표시
            for i, endpoint in si_endpoints.items():
                x, y = endpoint['boundary']
                ellipse = scene.addEllipse(x-5, y-5, 10, 10, QPen(Qt.green), QBrush(Qt.green))
                ellipse.setData(0, f"si_{i}")  # 식별자 저장
                
                text = scene.addText(f"Si {i}")
                text.setPos(x+10, y-10)
                text.setDefaultTextColor(Qt.green)
                text.setData(0, f"si_{i}")  # 식별자 저장
                
                si_point_items[i] = ellipse
            
            # SiGe 센터 표시
            for i, center in sige_centers.items():
                x, y = center
                ellipse = scene.addEllipse(x-5, y-5, 10, 10, QPen(Qt.red), QBrush(Qt.red))
                ellipse.setData(0, f"sige_{i}")  # 식별자 저장
                
                text = scene.addText(f"SiGe {i}")
                text.setPos(x+10, y-10)
                text.setDefaultTextColor(Qt.red)
                text.setData(0, f"sige_{i}")  # 식별자 저장
                
                sige_point_items[i] = ellipse
            
            # 컨트롤 패널
            control_panel = QWidget()
            control_layout = QVBoxLayout(control_panel)
            
            # Si 레이어 조정 그룹
            si_group = QGroupBox("Si Layer Endpoints")
            si_layout = QVBoxLayout()
            
            for i in si_endpoints.keys():
                si_row = QHBoxLayout()
                si_label = QLabel(f"Si Layer {i}:")
                si_row.addWidget(si_label)
                
                x_label = QLabel("X:")
                si_row.addWidget(x_label)
                x_spinner = QSpinBox()
                x_spinner.setRange(0, width)
                x_spinner.setValue(int(si_endpoints[i]['boundary'][0]))
                si_row.addWidget(x_spinner)
                
                y_label = QLabel("Y:")
                si_row.addWidget(y_label)
                y_spinner = QSpinBox()
                y_spinner.setRange(0, height)
                y_spinner.setValue(int(si_endpoints[i]['boundary'][1]))
                si_row.addWidget(y_spinner)
                
                si_spinners[i] = {'x': x_spinner, 'y': y_spinner}
                
                # X, Y 값 변경 시 점 이동
                def update_si_point(i=i):
                    x = si_spinners[i]['x'].value()
                    y = si_spinners[i]['y'].value()
                    si_point_items[i].setRect(x-5, y-5, 10, 10)
                    
                    # 텍스트 위치도 업데이트
                    for item in scene.items():
                        if isinstance(item, QGraphicsTextItem) and item.data(0) == f"si_{i}":
                            item.setPos(x+10, y-10)
                            
                    si_endpoints[i]['boundary'] = [x, y]
                    
                x_spinner.valueChanged.connect(update_si_point)
                y_spinner.valueChanged.connect(update_si_point)
                
                si_layout.addLayout(si_row)
                
            si_group.setLayout(si_layout)
            control_layout.addWidget(si_group)
            
            # SiGe 레이어 조정 그룹
            sige_group = QGroupBox("SiGe Layer Centers")
            sige_layout = QVBoxLayout()
            
            for i in sige_centers.keys():
                sige_row = QHBoxLayout()
                sige_label = QLabel(f"SiGe Layer {i}:")
                sige_row.addWidget(sige_label)
                
                x_label = QLabel("X:")
                sige_row.addWidget(x_label)
                x_spinner = QSpinBox()
                x_spinner.setRange(0, width)
                x_spinner.setValue(int(sige_centers[i][0]))
                sige_row.addWidget(x_spinner)
                
                y_label = QLabel("Y:")
                sige_row.addWidget(y_label)
                y_spinner = QSpinBox()
                y_spinner.setRange(0, height)
                y_spinner.setValue(int(sige_centers[i][1]))
                sige_row.addWidget(y_spinner)
                
                sige_spinners[i] = {'x': x_spinner, 'y': y_spinner}
                
                # X, Y 값 변경 시 점 이동
                def update_sige_point(i=i):
                    try:
                        x = sige_spinners[i]['x'].value()
                        y = sige_spinners[i]['y'].value()
                        sige_point_items[i].setRect(x-5, y-5, 10, 10)
                        
                        # 텍스트 위치도 업데이트
                        for item in scene.items():
                            if isinstance(item, QGraphicsTextItem) and item.data(0) == f"sige_{i}":
                                item.setPos(x+10, y-10)
                                
                        sige_centers[i] = [x, y]
                        
                    except KeyError:
                        print(f"Warning: SiGe spinner with index {i} not found")
                    except Exception as e:
                        print(f"Error updating SiGe point: {str(e)}")
                    
                x_spinner.valueChanged.connect(update_sige_point)
                y_spinner.valueChanged.connect(update_sige_point)
                
                sige_layout.addLayout(sige_row)
                
            sige_group.setLayout(sige_layout)
            control_layout.addWidget(sige_group)
            
            # 스크롤 영역에 컨트롤 패널 추가
            scroll_area = QScrollArea()
            scroll_area.setWidget(control_panel)
            scroll_area.setWidgetResizable(True)
            layout.addWidget(scroll_area, stretch=1)
            
            # 버튼
            button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            button_box.accepted.connect(adjust_dialog.accept)
            button_box.rejected.connect(adjust_dialog.reject)
            layout.addWidget(button_box)
            
            adjust_dialog.setLayout(layout)
            
            # 다이얼로그 실행 및 결과 처리
            if adjust_dialog.exec_() == QDialog.Accepted:
                # 조정된 포인트 저장
                self.adjusted_si_endpoints = {i: endpoint['boundary'] for i, endpoint in si_endpoints.items()}
                self.adjusted_sige_centers = sige_centers
                QMessageBox.information(self, "Success", "레이어 포인트가 조정되었습니다.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"레이어 포인트 조정 중 오류 발생: {str(e)}")
            traceback.print_exc()

    def calculate_selectivity(self):
        """다층 구조의 레이어 선택비 분석"""
        try:
            # 1. 필요한 라이브러리 임포트 확인
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy import ndimage
            import pandas as pd
            from io import BytesIO
            
            # 2. 데이터 준비
            # 현재 레이어 데이터와 마스크 수집
            layers_data = []
            masks = []
            
            for i, layer in enumerate(self.layers):
                if 'masks' in layer and len(layer['masks']) > 0:
                    layer_info = {
                        'name': layer['name'],
                        'layer_type': layer.get('layer_type', 'Si'),
                    }
                    layers_data.append(layer_info)
                    masks.append(layer['masks'][0])
                
            if len(layers_data) < 2:
                QMessageBox.warning(self, "Warning", "선택비 계산을 위해 최소 2개 이상의 레이어가 필요합니다.")
                return
                
            # 3. Si와 SiGe 레이어 분류
            si_layer = [i for i, layer_data in enumerate(layers_data) if layer_data['layer_type'] == 'Si']
            si_layer_mask = {i: masks[i] for i, layer_data in enumerate(layers_data) if layer_data['layer_type'] == 'Si'}
            
            sige_layer = [i for i, layer_data in enumerate(layers_data) if layer_data['layer_type'] == 'SiGe']
            sige_layer_mask = {i: masks[i] for i, layer_data in enumerate(layers_data) if layer_data['layer_type'] == 'SiGe'}
            
            if not si_layer or not sige_layer:
                QMessageBox.warning(self, "Warning", "선택비 계산을 위해 Si와 SiGe 타입의 레이어가 각각 필요합니다.")
                return
                
            # 4. 구조 방향 확인 (좌측 또는 우측 정렬)
            is_right_structure = self.determine_structure_position() == 'right'
            
            # 5. 조정된 포인트 불러오기
            adjusted_si_endpoints = getattr(self, 'adjusted_si_endpoints', {})
            adjusted_sige_centers = getattr(self, 'adjusted_sige_centers', {})
            
            # 6. 분석 함수 정의
            def find_intersection(line1, line2):
                """두 직선의 교차점 계산"""
                # 둘 다 수직선인 경우
                if line1.get('is_vertical', False) and line2.get('is_vertical', False):
                    return None
                
                # line1이 수직선인 경우
                if line1.get('is_vertical', False):
                    x = line1['x']
                    y = line2['slope'] * x + line2['intercept']
                    return [x, y]
                
                # line2가 수직선인 경우
                if line2.get('is_vertical', False):
                    x = line2['x']
                    y = line1['slope'] * x + line1['intercept']
                    return [x, y]
                
                # 둘 다 일반 직선인 경우
                if abs(line1['slope'] - line2['slope']) < 1e-10:
                    return None  # 평행한 경우
                
                # 교차점 계산
                x = (line2['intercept'] - line1['intercept']) / (line1['slope'] - line2['slope'])
                y = line1['slope'] * x + line1['intercept']
                
                return [x, y]
                
            def find_mask_boundary(mask):
                """마스크 경계 찾기"""
                eroded = ndimage.binary_erosion(mask)
                boundary = mask.astype(int) - eroded.astype(int)
                return np.where(boundary > 0)
                
            def point_to_line_distance(point, line):
                """점과 선 사이의 거리 계산"""
                if line.get('is_vertical', False):
                    return abs(point[0] - line['x'])
                
                # 직선 방정식: ax + by + c = 0 에서 a=slope, b=-1, c=intercept
                a = line['slope']
                b = -1
                c = line['intercept']
                
                # 점과 선 사이 거리 공식: |ax0 + by0 + c| / sqrt(a^2 + b^2)
                numerator = abs(a * point[0] + b * point[1] + c)
                denominator = np.sqrt(a**2 + b**2)
                
                return numerator / denominator
                
            def find_line_mask_intersection(line, mask):
                """라인과 마스크 경계의 교차점들 중 가장 아래쪽 점 찾기"""
                # 마스크 경계 찾기
                y_boundary, x_boundary = find_mask_boundary(mask)
                
                if len(x_boundary) == 0:
                    return None
                
                # 경계점들을 좌표 배열로 변환
                boundary_points = np.array(list(zip(x_boundary, y_boundary)))
                
                # 라인 범위 내에 있는 점들만 필터링
                in_range_points = []
                for point in boundary_points:
                    x, y = point
                    if (line.get('is_vertical', False) and 
                        line['x'] == x and 
                        line['y_min'] <= y <= line['y_max']):
                        in_range_points.append(point)
                    elif (not line.get('is_vertical', False) and
                        line['x_min'] <= x <= line['x_max'] and
                        line['y_min'] <= y <= line['y_max']):
                        in_range_points.append(point)
                
                if not in_range_points:
                    # 범위 제한을 조금 늘려서 다시 시도
                    for point in boundary_points:
                        x, y = point
                        if (line.get('is_vertical', False) and 
                            abs(line['x'] - x) < 2 and 
                            line['y_min'] - 2 <= y <= line['y_max'] + 2):
                            in_range_points.append(point)
                        elif (not line.get('is_vertical', False) and
                            line['x_min'] - 2 <= x <= line['x_max'] + 2 and
                            line['y_min'] - 2 <= y <= line['y_max'] + 2):
                            in_range_points.append(point)
                
                if not in_range_points:
                    return None
                
                # 각 점에서 라인까지의 거리 계산
                distances = []
                for point in in_range_points:
                    dist = point_to_line_distance(point, line)
                    distances.append(dist)
                
                # 라인에 가장 가까운 점들 찾기 (임계값 내)
                threshold = 1.5  # 픽셀 단위
                close_indices = [i for i, d in enumerate(distances) if d < threshold]
                
                if not close_indices:
                    return None
                    
                # 가장 가까운 점들 중에서 y 좌표가 가장 큰 점(가장 아래쪽) 선택
                closest_points = [in_range_points[i] for i in close_indices]
                y_coords = [point[1] for point in closest_points]
                bottom_idx = np.argmax(y_coords)  # y 좌표가 가장 큰 인덱스 선택
                
                return closest_points[bottom_idx]
            
            # 7. 분석 시작 - 마스크 오버레이 생성
            plt.figure(figsize=(15, 6))
                    
            plt.imshow(self.image, cmap="gray")
                
            # 8. Si 센터/경계점 계산
            si_center_point = {}
            si_regression_lines = {}
            si_endpoints = {}
            
            for i in si_layer:
                mask = masks[i]
                y_coords, x_coords = np.where(mask > 0)
                
                if len(x_coords) == 0:
                    continue
                    
                # Si Layer 끝점 계산
                left_x = np.min(x_coords)
                left_y = np.mean(y_coords[x_coords == left_x])
                right_x = np.max(x_coords)
                right_y = np.mean(y_coords[x_coords == right_x])
                
                si_endpoints[i] = {
                    'left': [left_x, left_y],
                    'right': [right_x, right_y]
                }
                
                # 조정된 경계점 사용 또는 계산
                if i in adjusted_si_endpoints:
                    si_center_point[i] = adjusted_si_endpoints[i]
                else:
                    # 구조에 따라 중요한 경계점 선택
                    if is_right_structure:
                        center_x = left_x  # 오른쪽 구조는 좌측 경계가 중요
                        center_y = left_y
                    else:
                        center_x = right_x  # 왼쪽 구조는 우측 경계가 중요
                        center_y = right_y
                        
                    si_center_point[i] = [center_x, center_y]
                
                # 회귀선 계산용 포인트 필터링
                sige_keys = list(sige_layer)
                si_idx = si_layer.index(i)
                if si_idx < len(sige_keys):
                    sige_idx = sige_keys[si_idx]
                    
                    # SiGe 센터 좌표 가져오기
                    if sige_idx in adjusted_sige_centers:
                        sige_x = adjusted_sige_centers[sige_idx][0]
                    elif sige_idx in sige_layer_mask:
                        # 기본 SiGe 센터 계산
                        sige_mask = sige_layer_mask[sige_idx]
                        sige_y_coords, sige_x_coords = np.where(sige_mask > 0)
                        center_y = int(np.median(sige_y_coords))
                        x_values_at_center_y = sige_x_coords[sige_y_coords == center_y]
                        
                        if is_right_structure:
                            sige_x = np.min(x_values_at_center_y)  # 오른쪽 구조
                        else:
                            sige_x = np.max(x_values_at_center_y)  # 왼쪽 구조
                    else:
                        continue
                    
                    # SiGe 센터 이전/이후의 포인트 필터링 (구조에 따라)
                    if is_right_structure:
                        filtered_x = np.unique(x_coords)[np.unique(x_coords) < sige_x]  # 오른쪽 구조
                    else:
                        filtered_x = np.unique(x_coords)[np.unique(x_coords) > sige_x]  # 왼쪽 구조
                        
                    filtered_y_means = [y_coords[x_coords == x_coord].mean() for x_coord in filtered_x]
                    
                    # 회귀분석 (최소 2개 이상의 점이 있어야 함)
                    if len(filtered_x) >= 2:
                        # 선형 회귀
                        coeff = np.polyfit(filtered_x, filtered_y_means, 1)
                        slope = coeff[0]      # 기울기
                        intercept = coeff[1]  # y절편
                        
                        # 기울기가 같고 SiGe 중심을 지나는 선 계산
                        if sige_idx in adjusted_sige_centers:
                            sige_center = adjusted_sige_centers[sige_idx]
                        else:
                            sige_center = [sige_x, center_y]
                            
                        parallel_intercept = sige_center[1] - slope * sige_center[0]
                        
                        # 회귀선 정보 저장
                        si_regression_lines[i] = {
                            'slope': slope,
                            'intercept': intercept,
                            'parallel_intercept': parallel_intercept
                        }
                
                # Si 센터 포인트 시각화
                plt.scatter(si_center_point[i][0], si_center_point[i][1], color='blue', s=30, marker='o')
                
            # 9. SiGe 센터 계산
            sige_center_point = {}
            for i in sige_layer:
                # 조정된 센터 사용 또는 계산
                if i in adjusted_sige_centers:
                    sige_center_point[i] = adjusted_sige_centers[i]
                else:
                    mask = masks[i]
                    y_coords, x_coords = np.where(mask > 0)
                    
                    if len(x_coords) == 0:
                        continue
                        
                    center_y = int(np.median(y_coords))
                    x_values_at_center_y = x_coords[y_coords == center_y]
                    
                    if is_right_structure:
                        center_x = np.min(x_values_at_center_y)  # 오른쪽 구조
                    else:
                        center_x = np.max(x_values_at_center_y)  # 왼쪽 구조
                        
                    sige_center_point[i] = [center_x, center_y]
                    
                # SiGe 센터 포인트 시각화    
                plt.scatter(sige_center_point[i][0], sige_center_point[i][1], color='red', s=30, marker='o')
                
            # 10. Si 끝점 연결 및 오프셋 라인 생성
            si_endpoint_connections = {}
            si_offset_lines = []
            sige_offset_lines = []
            
            x_pre = False
            y_pre = False
            prev_slope = None
            prev_intercept = None
            
            for i, [x, y] in si_center_point.items():
                if y_pre and x_pre:
                    # 빨간 연결선 그리기
                    # plt.plot([x_pre, x], [y_pre, y], color="red", linewidth=1)
                    
                    # 연결선 정보 저장
                    if x != x_pre:  # 수직선 방지
                        slope = (y - y_pre) / (x - x_pre)
                        intercept = y_pre - slope * x_pre
                        si_endpoint_connections[f"si_{i-1}_{i}"] = {
                            'points': [[x_pre, y_pre], [x, y]],
                            'x_min': min(x_pre, x),
                            'x_max': max(x_pre, x),
                            'slope': slope,
                            'intercept': intercept
                        }
                        # 이전 기울기와 절편 저장 (마지막 레이어 처리용)
                        prev_slope = slope
                        prev_intercept = intercept
                    else:
                        # 수직선 처리
                        si_endpoint_connections[f"si_{i-1}_{i}_vertical"] = {
                            'points': [[x_pre, y_pre], [x, y]],
                            'x': x,
                            'y_min': min(y_pre, y),
                            'y_max': max(y_pre, y),
                            'is_vertical': True
                        }
                    
                    # 오프셋된 연결선 그리기
                    offset = 10 // self.pixel_to_nm
                    if is_right_structure:
                        offset_x1, offset_x2 = x_pre + offset, x + offset  # 오른쪽 구조
                    else:
                        offset_x1, offset_x2 = x_pre - offset, x - offset  # 왼쪽 구조
                        
                    plt.plot([offset_x1, offset_x2], [y_pre, y], color="red", linewidth=1)
                    
                    # 오프셋 라인 정보 저장
                    offset_y1, offset_y2 = y_pre, y
                    
                    if offset_x1 != offset_x2:  # 수직선 방지
                        offset_slope = (offset_y2 - offset_y1) / (offset_x2 - offset_x1)
                        offset_intercept = offset_y1 - offset_slope * offset_x1
                        si_offset_lines.append({
                            'type': 'si',
                            'layer_idx': i,
                            'start': [offset_x1, offset_y1],
                            'end': [offset_x2, offset_y2],
                            'slope': offset_slope,
                            'intercept': offset_intercept,
                            'x_min': min(offset_x1, offset_x2),
                            'x_max': max(offset_x1, offset_x2),
                            'y_min': min(offset_y1, offset_y2),
                            'y_max': max(offset_y1, offset_y2)
                        })
                    else:
                        # 수직선 처리
                        si_offset_lines.append({
                            'type': 'si',
                            'layer_idx': i,
                            'start': [offset_x1, offset_y1],
                            'end': [offset_x2, offset_y2],
                            'is_vertical': True,
                            'x': offset_x1,
                            'y_min': min(offset_y1, offset_y2),
                            'y_max': max(offset_y1, offset_y2)
                        })
                        
                # 첫 번째 레이어인 경우 y값 감소 방향으로 연장
                elif i == min(si_center_point.keys()) and x and y:
                    # 첫 번째 레이어 정보 저장
                    first_si_x, first_si_y = x, y
                    
                    # 다음 레이어와의 연결선 기울기 사용 (다음 레이어가 없다면 회귀선 기울기 사용)
                    extension_slope = None
                    if i+1 in si_center_point:
                        next_x, next_y = si_center_point[i+1]
                        if next_x != x:  # 수직선 방지
                            extension_slope = (next_y - y) / (next_x - x)
                    
                    if extension_slope is None and i in si_regression_lines:
                        extension_slope = si_regression_lines[i]['slope']
                    
                    if extension_slope is not None:
                        # y값 감소 방향으로 연장
                        y_min = 0  # 이미지 상단
                        extended_x = x - (y - y_min) / extension_slope if extension_slope != 0 else x
                        
                        # 연장선 그리기
                        plt.plot([x, extended_x], [y, y_min], 
                                    color="red", linewidth=1, linestyle='--')
                        
                        # 연장선 정보 저장
                        si_endpoint_connections[f"first_extension_{i}"] = {
                            'points': [[x, y], [extended_x, y_min]],
                            'x_min': min(x, extended_x),
                            'x_max': max(x, extended_x),
                            'y_min': y_min,
                            'y_max': y,
                            'slope': extension_slope,
                            'intercept': y - extension_slope * x
                        }
                        
                        # 오프셋된 연장선 그리기
                        offset = 10 // self.pixel_to_nm
                        if is_right_structure:
                            offset_x1 = x + offset  # 오른쪽 구조
                            extended_offset_x = extended_x + offset
                        else:
                            offset_x1 = x - offset  # 왼쪽 구조
                            extended_offset_x = extended_x - offset
                            
                        plt.plot([offset_x1, extended_offset_x], 
                                    [y, y_min], 
                                    color="red", linewidth=1, linestyle='--')
                        
                        # 오프셋 연장선 정보 저장
                        si_offset_lines.append({
                            'type': 'si_first_extension',
                            'layer_idx': i,
                            'start': [offset_x1, y],
                            'end': [extended_offset_x, y_min],
                            'slope': extension_slope,
                            'intercept': y - extension_slope * offset_x1,
                            'x_min': min(offset_x1, extended_offset_x),
                            'x_max': max(offset_x1, extended_offset_x),
                            'y_min': y_min,
                            'y_max': y
                        })
                x_pre = x
                y_pre = y
                
            # 11. 마지막 Si 레이어 연장
            if prev_slope is not None and prev_intercept is not None:
                last_x, last_y = list(si_center_point.values())[-1]
                
                # 연장 지점 계산 (아래쪽으로 확장하기 위해 y값 증가)
                y_max = masks[0].shape[0]  # 이미지의 높이
                
                # 기울기가 0인 경우 처리
                if abs(prev_slope) < 1e-10:
                    if is_right_structure:
                        extended_x = last_x + 500  # 오른쪽으로 확장
                    else:
                        extended_x = last_x - 500  # 왼쪽으로 확장
                    extended_y = last_y
                else:
                    # 연장할 y 좌표 설정 (아래쪽으로 충분히 확장)
                    extended_y = y_max
                    # 해당 y 좌표에서의 x 좌표 계산
                    extended_x = (extended_y - prev_intercept) / prev_slope
                
                # 연장선 그리기
                plt.plot([last_x, extended_x], [last_y, extended_y], 
                            color="red", linewidth=1, linestyle='--')
                
                # 연장선 정보 저장
                last_idx = max(si_center_point.keys())
                si_endpoint_connections[f"last_extension_{last_idx}"] = {
                    'points': [[last_x, last_y], [extended_x, extended_y]],
                    'x_min': min(last_x, extended_x),
                    'x_max': max(last_x, extended_x),
                    'y_min': min(last_y, extended_y),
                    'y_max': max(last_y, extended_y),
                    'slope': prev_slope,
                    'intercept': prev_intercept
                }
                
                # 오프셋된 연장선 그리기
                offset = 10 // self.pixel_to_nm
                if is_right_structure:
                    offset_x1 = last_x + offset  # 오른쪽 구조
                    extended_offset_x = extended_x + offset
                else:
                    offset_x1 = last_x - offset  # 왼쪽 구조
                    extended_offset_x = extended_x - offset
                    
                plt.plot([offset_x1, extended_offset_x], 
                            [last_y, extended_y], color="red", linewidth=1, linestyle='--')
                
                # 오프셋 연장선 정보 저장
                offset_y1, offset_y2 = last_y, extended_y
                
                if offset_x1 != extended_offset_x:  # 수직선 방지
                    offset_slope = (offset_y2 - offset_y1) / (extended_offset_x - offset_x1)
                    offset_intercept = offset_y1 - offset_slope * offset_x1
                    si_offset_lines.append({
                        'type': 'si_extension',
                        'layer_idx': last_idx,
                        'start': [offset_x1, offset_y1],
                        'end': [extended_offset_x, offset_y2],
                        'slope': offset_slope,
                        'intercept': offset_intercept,
                        'x_min': min(offset_x1, extended_offset_x),
                        'x_max': max(offset_x1, extended_offset_x),
                        'y_min': min(offset_y1, offset_y2),
                        'y_max': max(offset_y1, offset_y2)
                    })
                else:
                    # 수직선 처리
                    si_offset_lines.append({
                        'type': 'si_extension',
                        'layer_idx': last_idx,
                        'start': [offset_x1, offset_y1],
                        'end': [extended_offset_x, offset_y2],
                        'is_vertical': True,
                        'x': offset_x1,
                        'y_min': min(offset_y1, offset_y2),
                        'y_max': max(offset_y1, offset_y2)
                    })
            
            # 12. 교차점 및 거리 계산
            intersection_results = {}
            
            for i, reg_info in si_regression_lines.items():
                sige_keys = list(sige_center_point.keys())
                si_idx = list(si_center_point.keys()).index(i)
                if si_idx < len(sige_keys):
                    sige_idx = sige_keys[si_idx]
                    sige_center = sige_center_point[sige_idx]
                    
                    # SiGe 중심을 지나는 평행선
                    parallel_line = {
                        'slope': reg_info['slope'],
                        'intercept': reg_info['parallel_intercept']
                    }
                    
                    # 해당 SiGe 레이어와 연관된 Si 연결선 찾기
                    best_intersection = None
                    min_distance = float('inf')
                    best_connection = None
                    
                    # 모든 Si 연결선과 교차점 계산
                    for conn_key, conn in si_endpoint_connections.items():
                        intersection = find_intersection(parallel_line, conn)
                        
                        if intersection:
                            # 교차점이 연결선 범위 내에 있는지 확인
                            in_range = False
                            if conn.get('is_vertical', False):
                                if conn['y_min'] <= intersection[1] <= conn['y_max']:
                                    in_range = True
                            else:
                                if conn['x_min'] <= intersection[0] <= conn['x_max']:
                                    in_range = True
                            
                            if in_range:
                                # 교차점과 SiGe 중심 사이의 거리 계산
                                distance = np.sqrt((sige_center[0] - intersection[0])**2 + 
                                                    (sige_center[1] - intersection[1])**2)
                                
                                # 가장 가까운 교차점 찾기
                                if distance < min_distance:
                                    min_distance = distance
                                    best_intersection = intersection
                                    best_connection = conn_key
                    
                    if best_intersection:
                        # 교차점 표시
                        plt.scatter(best_intersection[0], best_intersection[1], color='magenta', s=30, marker='x')
                        
                        # 거리 표시 라인
                        plt.plot([sige_center[0], best_intersection[0]], 
                                    [sige_center[1], best_intersection[1]], 
                                    'magenta', linestyle='-.', linewidth=1)
                        
                        # 거리 텍스트
                        mid_x = (sige_center[0] + best_intersection[0]) / 2
                        mid_y = (sige_center[1] + best_intersection[1]) / 2
                        plt.text(mid_x, mid_y - 10, f"{min_distance * self.pixel_to_nm:.2f}nm", 
                                    color='magenta', fontsize=9, ha='center',
                                    bbox=dict(facecolor='black', alpha=0.6))
                        
                        # 결과 저장
                        intersection_results[sige_idx] = {
                            'si_layer': i,
                            'regression_slope': reg_info['slope'],
                            'connection_line': best_connection,
                            'intersection': best_intersection,
                            'distance': min_distance,
                            'SiGe_Recess': min_distance * self.pixel_to_nm
                        }
            
            # 13. SiGe 센터 연결선 그리기
            x_pre = False
            y_pre = False
            first_sige_slope = None
            
            for i, [x, y] in sige_center_point.items():
                # 오프셋 계산
                offset = 50 // self.pixel_to_nm
                if is_right_structure:
                    offset_x = x + offset  # 오른쪽 구조
                else:
                    offset_x = x - offset  # 왼쪽 구조
                    
                plt.scatter(offset_x, y, color='red', s=3, marker="o")
                if y_pre and x_pre and x_pre_offset:
                    # 연결선 그리기
                    # plt.plot([x_pre, x], [y_pre, y], color="green", linewidth=0.5)
                    
                    # 오프셋 연결선 그리기
                    # plt.plot([x_pre_offset, offset_x], [y_pre, y], color="green", linewidth=0.5)
                    
                    # 오프셋 라인 정보 저장
                    if x_pre_offset != offset_x:  # 수직선 방지
                        offset_slope = (y - y_pre) / (offset_x - x_pre_offset)
                        offset_intercept = y_pre - offset_slope * x_pre_offset
                        sige_offset_lines.append({
                            'type': 'sige',
                            'layer_idx': i,
                            'start': [x_pre_offset, y_pre],
                            'end': [offset_x, y],
                            'slope': offset_slope,
                            'intercept': offset_intercept,
                            'x_min': min(x_pre_offset, offset_x),
                            'x_max': max(x_pre_offset, offset_x),
                            'y_min': min(y_pre, y),
                            'y_max': max(y_pre, y)
                        })
                        
                        # 첫 번째 SiGe 연결선의 기울기 저장
                        if first_sige_slope is None:
                            first_sige_slope = offset_slope
                    else:
                        # 수직선 처리
                        sige_offset_lines.append({
                            'type': 'sige',
                            'layer_idx': i,
                            'start': [x_pre_offset, y_pre],
                            'end': [offset_x, y],
                            'is_vertical': True,
                            'x': offset_x,
                            'y_min': min(y_pre, y),
                            'y_max': max(y_pre, y)
                        })
                # 첫 번째 SiGe 레이어인 경우 y값 감소 방향으로 연장
                elif i == min(sige_center_point.keys()) and x and y:
                    # 첫 번째 레이어 정보
                    first_sige_x, first_sige_y = x, y
                    
                    # 다음 레이어와의 연결선 기울기 구하기
                    extension_slope = None
                    if i+1 in sige_center_point:
                        next_x, next_y = sige_center_point[i+1]
                        if next_x != x:  # 수직선 방지
                            extension_slope = (next_y - y) / (next_x - x)
                    
                    # 연결된 Si 레이어의 회귀선 기울기 사용
                    si_idx = list(si_layer_mask.keys())[list(sige_center_point.keys()).index(i)]
                    if extension_slope is None and si_idx in si_regression_lines:
                        extension_slope = si_regression_lines[si_idx]['slope']
                    
                    if extension_slope is not None:
                        # y값 감소 방향으로 연장
                        y_min = 0  # 이미지 상단
                        if is_right_structure:
                            extended_x = first_sige_x - (first_sige_y - y_min) / extension_slope if extension_slope != 0 else first_sige_x
                            extended_offset_x = extended_x + offset
                        else:
                            extended_x = first_sige_x - (first_sige_y - y_min) / extension_slope if extension_slope != 0 else first_sige_x
                            extended_offset_x = extended_x - offset
                        
                        # # 연장된 오프셋 라인 그리기
                        # plt.plot([offset_x, extended_offset_x], 
                        #             [y, y_min], 
                        #             color="red", linewidth=1, linestyle='--')
                        
                        # 오프셋 연장선 정보 저장
                        sige_offset_lines.append({
                            'type': 'sige_first_extension',
                            'layer_idx': i,
                            'start': [offset_x, y],
                            'end': [extended_offset_x, y_min],
                            'slope': extension_slope,
                            'intercept': y - extension_slope * offset_x,
                            'x_min': min(offset_x, extended_offset_x),
                            'x_max': max(offset_x, extended_offset_x),
                            'y_min': y_min,
                            'y_max': y
                        })
                x_pre = x
                y_pre = y
                x_pre_offset = offset_x
                    
            # 14. Si 오프셋 라인과 레이어 마스크 교차점 찾기 및 두께 측정
            si_intersections = []
            post_si_thickness_measurements = []
            
            for check_idx, line in enumerate(si_offset_lines):
                # 현재 레이어와의 교차점 확인
                if check_idx not in si_layer_mask:
                    continue
                    
                intersection = find_line_mask_intersection(line, si_layer_mask[check_idx])
                
                if intersection is not None:
                    x, y = intersection
                    si_intersections.append({
                        'line': line,
                        'intersection': intersection,
                        'line_layer': line['layer_idx'],
                        'mask_layer': check_idx,
                        'x': x,
                        'y': y
                    })
                    
                    # 교차점 시각화
                    plt.scatter(x, y, color='lime', s=5, marker='+')
                    
                    # 교차점에서 두께 측정
                    thickness = self.measure_thickness_at_intersection(si_layer_mask[check_idx], x, y)
                    if thickness:
                        post_si_thickness_measurements.append({
                            'layer': check_idx,
                            'x': x,
                            'y': y,
                            'thickness': thickness['thickness'],
                            'thickness_nm': thickness['thickness_nm'],
                            'start_point': thickness.get('start_point'),
                            'end_point': thickness.get('end_point')
                        })
                            
                        # 두께 측정 시각화
                        start_x, start_y = thickness['start_point']
                        end_x, end_y = thickness['end_point']
                        
                        # 측정선 그리기
                        plt.plot([start_x, end_x], [start_y, end_y], 'g-', linewidth=2)
                        
                        # 두께 텍스트 표시
                        mid_x = (start_x + end_x) / 2
                        mid_y = (start_y + end_y) / 2
                        plt.text(mid_x + 10, mid_y, f"{thickness['thickness_nm']:.2f}nm",
                                color='lime', fontsize=8,
                                bbox=dict(facecolor='black', alpha=0.6))
                        
                        # 접선 표시
                        if not thickness.get('is_vertical', False) and 'tangent_points' in thickness:
                            t1, t2 = thickness['tangent_points']
                            plt.plot([t1[0], t2[0]], [t1[1], t2[1]], 'g--', linewidth=1)

            
            # 15. SiGe 오프셋 라인과 레이어 마스크 교차점 찾기
            sige_intersections = []
            pre_si_thickness_measurements = []

            for line in sige_offset_lines:
                layer_idx = line['layer_idx']
                for check_idx in si_layer_mask.keys():
                    # 현재 레이어와의 교차점 확인
                    intersection = find_line_mask_intersection(line, si_layer_mask[check_idx])
                    
                    if intersection is not None:
                        x, y = intersection
                        sige_intersections.append({
                            'line': line,
                            'intersection': intersection,
                            'line_layer': layer_idx,
                            'mask_layer': check_idx,
                            'x': x,
                            'y': y
                        })
                        
                        # 교차점 시각화
                        plt.scatter(x, y, color='yellow', s=3, marker='*')

                        # 두께 측정
                        thickness = self.measure_thickness_at_intersection(si_layer_mask[check_idx], x, y)
                        if thickness:
                            pre_si_thickness_measurements.append({
                                'layer': check_idx,
                                'x': x,
                                'y': y,
                                'thickness': thickness['thickness'],
                                'thickness_nm': thickness['thickness_nm'],
                                'start_point': thickness.get('start_point'),
                                'end_point': thickness.get('end_point')
                            })
                            
                            # 두께 측정 시각화
                            start_x, start_y = thickness['start_point']
                            end_x, end_y = thickness['end_point']
                            
                            # 측정선 그리기
                            plt.plot([start_x, end_x], [start_y, end_y], 'y-', linewidth=0.5)
                            
                            # 두께 텍스트 표시
                            mid_x = (start_x + end_x) / 2
                            mid_y = (start_y + end_y) / 2
                            plt.text(mid_x + 10, mid_y, f"{thickness['thickness_nm']:.2f}nm",
                                    color='yellow', fontsize=6,
                                    bbox=dict(facecolor='black', alpha=0.3))
                
            # 16. 결과 마무리 및 시각화
            plt.tight_layout()
            plt.xlim(0, masks[0].shape[1])
            plt.ylim(masks[0].shape[0], 0)  # y축 반전
            
            # 그래프를 이미지로 저장
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150)
            buf.seek(0)
            plt.close()
            
            self.selectivity_data = {
                'structure_type': 'right' if is_right_structure else 'left',
                'si_layers': si_layer,
                'sige_layers': sige_layer,
                'si_center_points': si_center_point,
                'sige_center_points': sige_center_point,
                'intersection_results': intersection_results,
                'si_intersections': si_intersections,
                'sige_intersections': sige_intersections,
                'post_si_thickness_measurements': post_si_thickness_measurements,
                'pre_si_thickness_measurements': pre_si_thickness_measurements,
                'plot': buf
            }
            print("Selectivity data saved:", self.selectivity_data)
            self.display_image()
            # 18. 결과 다이얼로그 표시
            self.show_selectivity_results(self.selectivity_data)
        
        except Exception as e:
            QMessageBox.critical(self, "Error", f"선택비 계산 중 오류 발생: {str(e)}")
            traceback.print_exc()

    # def show_selectivity_results(self, data):
    #     """선택비 계산 결과 시각화 및 표시"""
    #     try:
    #         result_dialog = QDialog(self)
    #         result_dialog.setWindowTitle("Selectivity Analysis Results")
    #         result_dialog.setMinimumSize(1000, 800)
            
    #         # 전체 레이아웃
    #         main_layout = QVBoxLayout()
            
    #         # 스타일 설정
    #         title_style = "font-size: 18pt; font-weight: bold; color: #003366;"
    #         subtitle_style = "font-size: 14pt; font-weight: bold; color: #004d99; margin-top: 10px;"
    #         info_style = "font-size: 12pt; color: #555555;"
            
    #         # 상단 헤더 영역
    #         header_widget = QWidget()
    #         header_layout = QHBoxLayout(header_widget)
            
    #         # 로고 또는 아이콘 (플레이스홀더)
    #         icon_label = QLabel()
    #         icon_label.setFixedSize(80, 80)
    #         icon_label.setStyleSheet("background-color: #003366; border-radius: 40px;")
    #         header_layout.addWidget(icon_label)
            
    #         # 제목 및 정보
    #         title_widget = QWidget()
    #         title_layout = QVBoxLayout(title_widget)
            
    #         title_label = QLabel("Layer Selectivity Analysis")
    #         title_label.setStyleSheet(title_style)
    #         title_layout.addWidget(title_label)
            
    #         structure_label = QLabel(f"Structure Type: {'Right-aligned' if data['structure_type'] == 'right' else 'Left-aligned'}")
    #         structure_label.setStyleSheet(info_style)
    #         title_layout.addWidget(structure_label)
            
    #         date_label = QLabel(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    #         date_label.setStyleSheet(info_style)
    #         title_layout.addWidget(date_label)
            
    #         title_layout.setContentsMargins(0, 0, 0, 0)
    #         header_layout.addWidget(title_widget, 1)
            
    #         main_layout.addWidget(header_widget)
            
    #         # 구분선
    #         line = QFrame()
    #         line.setFrameShape(QFrame.HLine)
    #         line.setFrameShadow(QFrame.Sunken)
    #         line.setStyleSheet("background-color: #cccccc;")
    #         main_layout.addWidget(line)
            
    #         # 탭 위젯 생성 및 스타일 설정
    #         tab_widget = QTabWidget()
    #         tab_widget.setStyleSheet("""
    #             QTabWidget::pane {
    #                 border: 1px solid #cccccc;
    #                 background: white;
    #             }
    #             QTabBar::tab {
    #                 background: #e6e6e6;
    #                 border: 1px solid #cccccc;
    #                 padding: 6px 12px;
    #                 margin-right: 2px;
    #             }
    #             QTabBar::tab:selected {
    #                 background: #003366;
    #                 color: white;
    #             }
    #             QTabBar::tab:!selected {
    #                 background: #f0f0f0;
    #             }
    #             QTabBar::tab:hover {
    #                 background: #0055aa;
    #                 color: white;
    #             }
    #         """)
            
    #         # 1. 시각화 탭
    #         visualization_tab = QWidget()
    #         vis_layout = QVBoxLayout(visualization_tab)
            
    #         # 시각화 이미지
    #         image_label = QLabel()
    #         pixmap = QPixmap()
    #         pixmap.loadFromData(data['plot'].getvalue())
    #         image_label.setPixmap(pixmap)
    #         image_label.setScaledContents(False)
    #         image_label.setAlignment(Qt.AlignCenter)
            
    #         # 스크롤 영역에 이미지 추가
    #         scroll_area = QScrollArea()
    #         scroll_area.setWidget(image_label)
    #         scroll_area.setWidgetResizable(True)
    #         scroll_area.setStyleSheet("background-color: white; border: none;")
    #         vis_layout.addWidget(scroll_area)
            
    #         # 이미지 설명
    #         desc_label = QLabel("Layer boundaries, intersection points, and thickness measurements are visualized above.")
    #         desc_label.setStyleSheet(info_style)
    #         desc_label.setAlignment(Qt.AlignCenter)
    #         vis_layout.addWidget(desc_label)
            
    #         # 범례 생성
    #         legend_widget = QWidget()
    #         legend_layout = QHBoxLayout(legend_widget)
    #         legend_layout.setContentsMargins(20, 5, 20, 5)
            
    #         legend_items = [
    #             ("Si Center Points", "blue", "circle"), 
    #             ("SiGe Center Points", "red", "circle"),
    #             ("Si Intersection", "lime", "plus"),
    #             ("SiGe Intersection", "yellow", "star"),
    #             ("Distance", "magenta", "line")
    #         ]
            
    #         for text, color, shape in legend_items:
    #             item_widget = QWidget()
    #             item_layout = QHBoxLayout(item_widget)
    #             item_layout.setContentsMargins(0, 0, 0, 0)
                
    #             icon_label = QLabel()
    #             icon_label.setFixedSize(20, 20)
    #             if shape == "circle":
    #                 icon_label.setStyleSheet(f"background-color: {color}; border-radius: 10px;")
    #             elif shape == "plus":
    #                 icon_label.setStyleSheet(f"background-color: white; color: {color}; font-weight: bold; font-size: 16px; text-align: center;")
    #                 icon_label.setText("+")
    #             elif shape == "star":
    #                 icon_label.setStyleSheet(f"background-color: white; color: {color}; font-weight: bold; font-size: 16px; text-align: center;")
    #                 icon_label.setText("*")
    #             elif shape == "line":
    #                 icon_label.setStyleSheet(f"background-color: white; border-bottom: 2px {color} dashed;")
                
    #             item_layout.addWidget(icon_label)
                
    #             text_label = QLabel(text)
    #             text_label.setStyleSheet("color: #333333;")
    #             item_layout.addWidget(text_label)
                
    #             legend_layout.addWidget(item_widget)
            
    #         legend_layout.addStretch()
    #         vis_layout.addWidget(legend_widget)
            
    #         tab_widget.addTab(visualization_tab, "Visualization")
            
    #         # 2. 측정 결과 탭
    #         measurements_tab = QWidget()
    #         measurements_layout = QVBoxLayout(measurements_tab)
            
    #         # 두께 측정 테이블
    #         thickness_group = QGroupBox("Thickness Measurements")
    #         thickness_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #cccccc; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
    #         thickness_layout = QVBoxLayout(thickness_group)
            
    #         # 두께 측정 테이블
    #         thickness_table = QTableWidget()
    #         thick_headers = ["Type", "Layer", "X", "Y", "Thickness (nm)"]
    #         thickness_table.setColumnCount(len(thick_headers))
    #         thickness_table.setHorizontalHeaderLabels(thick_headers)
            
    #         # 모든 두께 측정 결과 결합
    #         all_thickness = []
    #         for item in data['post_si_thickness_measurements']:
    #             all_thickness.append({
    #                 'type': 'Si',
    #                 'layer': item['layer'],
    #                 'x': item['x'],
    #                 'y': item['y'],
    #                 'thickness_nm': item['thickness_nm']
    #             })
                
    #         for item in data['pre_si_thickness_measurements']:
    #             all_thickness.append({
    #                 'type': 'SiGe',
    #                 'layer': item['layer'],
    #                 'x': item['x'],
    #                 'y': item['y'],
    #                 'thickness_nm': item['thickness_nm']
    #             })
            
    #         # 데이터 추가
    #         thickness_table.setRowCount(len(all_thickness))
    #         for i, item in enumerate(all_thickness):
    #             thickness_table.setItem(i, 0, QTableWidgetItem(item['type']))
    #             thickness_table.setItem(i, 1, QTableWidgetItem(f"Layer {item['layer']}"))
    #             thickness_table.setItem(i, 2, QTableWidgetItem(f"{item['x']:.2f}"))
    #             thickness_table.setItem(i, 3, QTableWidgetItem(f"{item['y']:.2f}"))
    #             thickness_table.setItem(i, 4, QTableWidgetItem(f"{item['thickness_nm']:.2f}"))
                
    #             # Si와 SiGe에 따른 행 색상 구분
    #             if item['type'] == 'Si':
    #                 for j in range(thickness_table.columnCount()):
    #                     thickness_table.item(i, j).setBackground(QColor(240, 248, 255))  # Alice Blue
    #             else:
    #                 for j in range(thickness_table.columnCount()):
    #                     thickness_table.item(i, j).setBackground(QColor(255, 240, 245))  # Lavender Blush
            
    #         # 테이블 스타일
    #         thickness_table.setStyleSheet("""
    #             QTableWidget {
    #                 gridline-color: #d4d4d4;
    #                 background-color: white;
    #                 border: 1px solid #cccccc;
    #                 border-radius: 0px;
    #                 font: 10pt;
    #             }
    #             QTableWidget::item {
    #                 border-bottom: 1px solid #d4d4d4;
    #                 padding-left: 5px;
    #                 padding-right: 5px;
    #             }
    #             QHeaderView::section {
    #                 background-color: #003366;
    #                 color: white;
    #                 padding: 5px;
    #                 border: 1px solid #003366;
    #             }
    #         """)
    #         thickness_table.setEditTriggers(QTableWidget.NoEditTriggers)
    #         thickness_table.setAlternatingRowColors(True)
    #         thickness_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    #         thickness_table.verticalHeader().setVisible(False)
    #         thickness_table.setSelectionBehavior(QTableWidget.SelectRows)
            
    #         thickness_layout.addWidget(thickness_table)
    #         measurements_layout.addWidget(thickness_group)
            
    #         # 거리 측정 그룹
    #         distance_group = QGroupBox("SiGe-Si Distance Measurements")
    #         distance_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #cccccc; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
    #         distance_layout = QVBoxLayout(distance_group)
            
    #         # 거리 측정 테이블
    #         distance_table = QTableWidget()
    #         dist_headers = ["SiGe Layer", "Si Layer", "Distance (nm)", "Regression Slope"]
    #         distance_table.setColumnCount(len(dist_headers))
    #         distance_table.setHorizontalHeaderLabels(dist_headers)
            
    #         # 데이터 추가
    #         distance_table.setRowCount(len(data['intersection_results']))
    #         for i, (sige_idx, result) in enumerate(data['intersection_results'].items()):
    #             distance_table.setItem(i, 0, QTableWidgetItem(f"Layer {sige_idx}"))
    #             distance_table.setItem(i, 1, QTableWidgetItem(f"Layer {result['si_layer']}"))
    #             distance_table.setItem(i, 2, QTableWidgetItem(f"{result['SiGe_Recess']:.2f}"))
    #             distance_table.setItem(i, 3, QTableWidgetItem(f"{result['regression_slope']:.6f}"))
            
    #         # 테이블 스타일
    #         distance_table.setStyleSheet("""
    #             QTableWidget {
    #                 gridline-color: #d4d4d4;
    #                 background-color: white;
    #                 border: 1px solid #cccccc;
    #                 border-radius: 0px;
    #                 font: 10pt;
    #             }
    #             QTableWidget::item {
    #                 border-bottom: 1px solid #d4d4d4;
    #                 padding-left: 5px;
    #                 padding-right: 5px;
    #             }
    #             QHeaderView::section {
    #                 background-color: #003366;
    #                 color: white;
    #                 padding: 5px;
    #                 border: 1px solid #003366;
    #             }
    #         """)
    #         distance_table.setEditTriggers(QTableWidget.NoEditTriggers)
    #         distance_table.setAlternatingRowColors(True)
    #         distance_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    #         distance_table.verticalHeader().setVisible(False)
    #         distance_table.setSelectionBehavior(QTableWidget.SelectRows)
            
    #         distance_layout.addWidget(distance_table)
    #         measurements_layout.addWidget(distance_group)
            
    #         tab_widget.addTab(measurements_tab, "Measurements")
            
    #         # 3. 선택비 계산 탭
    #         selectivity_tab = QWidget()
    #         sel_layout = QVBoxLayout(selectivity_tab)
            
    #         # 선택비 계산 설명
    #         sel_header = QLabel("Layer Selectivity Calculator")
    #         sel_header.setStyleSheet(subtitle_style)
    #         sel_header.setAlignment(Qt.AlignCenter)
    #         sel_layout.addWidget(sel_header)
            
    #         sel_info = QLabel("Calculate selectivity ratio between any two layers. Selectivity is computed as the ratio of layer thicknesses.")
    #         sel_info.setStyleSheet(info_style)
    #         sel_info.setAlignment(Qt.AlignCenter)
    #         sel_info.setWordWrap(True)
    #         sel_layout.addWidget(sel_info)
    #         sel_layout.addSpacing(20)
            
    #         # 선택비 계산 UI
    #         calc_form = QWidget()
    #         calc_layout = QHBoxLayout(calc_form)
    #         calc_layout.setContentsMargins(50, 10, 50, 10)
            
    #         # 레이어 1 (분자)
    #         layer1_group = QGroupBox("Numerator Layer")
    #         layer1_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #cccccc; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
    #         layer1_layout = QVBoxLayout(layer1_group)
            
    #         layer_combo1 = QComboBox()
    #         layer_combo1.setStyleSheet("""
    #             QComboBox {
    #                 border: 1px solid #cccccc;
    #                 border-radius: 3px;
    #                 padding: 5px;
    #                 background-color: white;
    #             }
    #             QComboBox::drop-down {
    #                 border: 0px;
    #             }
    #             QComboBox::down-arrow {
    #                 image: url(dropdown.png);
    #                 width: 14px;
    #                 height: 14px;
    #             }
    #         """)
            
    #         for i in range(len(self.layers)):
    #             layer_type = self.layers[i].get('layer_type', 'Si')
    #             layer_combo1.addItem(f"Layer {i} ({layer_type})")
                
    #         layer1_layout.addWidget(layer_combo1)
    #         calc_layout.addWidget(layer1_group)
            
    #         # 나누기 기호
    #         div_label = QLabel("/")
    #         div_label.setStyleSheet("font-size: 24pt; font-weight: bold;")
    #         div_label.setAlignment(Qt.AlignCenter)
    #         calc_layout.addWidget(div_label)
            
    #         # 레이어 2 (분모)
    #         layer2_group = QGroupBox("Denominator Layer")
    #         layer2_group.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #cccccc; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
    #         layer2_layout = QVBoxLayout(layer2_group)
            
    #         layer_combo2 = QComboBox()
    #         layer_combo2.setStyleSheet("""
    #             QComboBox {
    #                 border: 1px solid #cccccc;
    #                 border-radius: 3px;
    #                 padding: 5px;
    #                 background-color: white;
    #             }
    #             QComboBox::drop-down {
    #                 border: 0px;
    #             }
    #             QComboBox::down-arrow {
    #                 image: url(dropdown.png);
    #                 width: 14px;
    #                 height: 14px;
    #             }
    #         """)
            
    #         for i in range(len(self.layers)):
    #             layer_type = self.layers[i].get('layer_type', 'Si')
    #             layer_combo2.addItem(f"Layer {i} ({layer_type})")
                
    #         if layer_combo2.count() > 1:
    #             layer_combo2.setCurrentIndex(1)
                
    #         layer2_layout.addWidget(layer_combo2)
    #         calc_layout.addWidget(layer2_group)
            
    #         sel_layout.addWidget(calc_form)
            
    #         # 계산 버튼
    #         calc_button = QPushButton("Calculate Selectivity")
    #         calc_button.setStyleSheet("""
    #             QPushButton {
    #                 background-color: #003366;
    #                 color: white;
    #                 padding: 8px;
    #                 font-size: 12pt;
    #                 border-radius: 4px;
    #                 min-width: 200px;
    #             }
    #             QPushButton:hover {
    #                 background-color: #004080;
    #             }
    #             QPushButton:pressed {
    #                 background-color: #002855;
    #             }
    #         """)
    #         calc_button.setFixedWidth(200)
    #         button_container = QWidget()
    #         button_layout = QHBoxLayout(button_container)
    #         button_layout.addStretch()
    #         button_layout.addWidget(calc_button)
    #         button_layout.addStretch()
    #         sel_layout.addWidget(button_container)
    #         sel_layout.addSpacing(10)
            
    #         # 결과 프레임
    #         result_frame = QFrame()
    #         result_frame.setFrameShape(QFrame.StyledPanel)
    #         result_frame.setStyleSheet("""
    #             QFrame {
    #                 border: 1px solid #cccccc;
    #                 border-radius: 5px;
    #                 background-color: #f9f9f9;
    #             }
    #         """)
    #         result_layout = QVBoxLayout(result_frame)
            
    #         # 결과 표시 레이블
    #         sel_result_label = QLabel("Select layers and click Calculate")
    #         sel_result_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #003366; min-height: 100px;")
    #         sel_result_label.setAlignment(Qt.AlignCenter)
    #         result_layout.addWidget(sel_result_label)
            
    #         sel_layout.addWidget(result_frame)
    #         sel_layout.addStretch()
            
    #         # 계산 버튼 클릭 이벤트
    #         def calculate_layer_selectivity():
    #             try:
    #                 layer1_idx = layer_combo1.currentIndex()
    #                 layer2_idx = layer_combo2.currentIndex()
                    
    #                 if layer1_idx == layer2_idx:
    #                     sel_result_label.setText("Please select different layers")
    #                     result_frame.setStyleSheet("QFrame { border: 1px solid #ffcccc; border-radius: 5px; background-color: #fff5f5; }")
    #                     return
                        
    #                 # 각 레이어의 두께 계산
    #                 thickness1 = self.calculate_average_thickness(self.layers[layer1_idx]['masks'][0])
    #                 thickness2 = self.calculate_average_thickness(self.layers[layer2_idx]['masks'][0])
                    
    #                 if thickness1 and thickness2:
    #                     selectivity = thickness1['average'] / thickness2['average']
    #                     layer1_type = self.layers[layer1_idx].get('layer_type', 'Si')
    #                     layer2_type = self.layers[layer2_idx].get('layer_type', 'Si')
                        
    #                     result_text = f"Selectivity Ratio: {selectivity:.2f}\n\n"
    #                     result_text += f"{layer1_type} Layer {layer1_idx}: {thickness1['average']:.2f} nm\n"
    #                     result_text += f"{layer2_type} Layer {layer2_idx}: {thickness2['average']:.2f} nm"
                        
    #                     sel_result_label.setText(result_text)
    #                     result_frame.setStyleSheet("QFrame { border: 1px solid #ccffcc; border-radius: 5px; background-color: #f0fff0; }")
    #                 else:
    #                     sel_result_label.setText("Could not calculate thickness for selected layers")
    #                     result_frame.setStyleSheet("QFrame { border: 1px solid #ffcccc; border-radius: 5px; background-color: #fff5f5; }")
    #             except Exception as e:
    #                 sel_result_label.setText(f"Error: {str(e)}")
    #                 result_frame.setStyleSheet("QFrame { border: 1px solid #ffcccc; border-radius: 5px; background-color: #fff5f5; }")
                    
    #         calc_button.clicked.connect(calculate_layer_selectivity)
            
    #         tab_widget.addTab(selectivity_tab, "Selectivity Calculator")
            
    #         # 4. 요약 탭 (대시보드 스타일)
    #         summary_tab = QWidget()
    #         summary_layout = QVBoxLayout(summary_tab)
            
    #         # 요약 스크롤 영역
    #         summary_scroll = QScrollArea()
    #         summary_scroll.setWidgetResizable(True)
    #         summary_scroll.setStyleSheet("background-color: white; border: none;")
    #         summary_container = QWidget()
    #         summary_container_layout = QVBoxLayout(summary_container)
    #         summary_container_layout.setContentsMargins(20, 20, 20, 20)
            
    #         # 대시보드 헤더
    #         dashboard_header = QLabel("Analysis Dashboard")
    #         dashboard_header.setStyleSheet(subtitle_style)
    #         dashboard_header.setAlignment(Qt.AlignCenter)
    #         summary_container_layout.addWidget(dashboard_header)
            
    #         # 구조 정보 카드
    #         structure_card = QFrame()
    #         structure_card.setFrameShape(QFrame.StyledPanel)
    #         structure_card.setStyleSheet("QFrame { background-color: white; border: 1px solid #dddddd; border-radius: 5px; }")
    #         structure_card_layout = QVBoxLayout(structure_card)
            
    #         card_title = QLabel("Structure Information")
    #         card_title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #003366;")
    #         structure_card_layout.addWidget(card_title)
            
    #         # 구조 정보 테이블
    #         structure_table = QTableWidget()
    #         structure_table.setColumnCount(2)
    #         structure_table.setHorizontalHeaderLabels(["Property", "Value"])
    #         structure_table.setRowCount(3)
            
    #         structure_table.setItem(0, 0, QTableWidgetItem("Structure Alignment"))
    #         structure_table.setItem(0, 1, QTableWidgetItem("Right-aligned" if data['structure_type'] == 'right' else "Left-aligned"))
            
    #         structure_table.setItem(1, 0, QTableWidgetItem("Si Layers"))
    #         structure_table.setItem(1, 1, QTableWidgetItem(str(len(data['si_layers']))))
            
    #         structure_table.setItem(2, 0, QTableWidgetItem("SiGe Layers"))
    #         structure_table.setItem(2, 1, QTableWidgetItem(str(len(data['sige_layers']))))
            
    #         # 테이블 스타일
    #         structure_table.setStyleSheet("""
    #             QTableWidget {
    #                 gridline-color: #d4d4d4;
    #                 background-color: white;
    #                 border: none;
    #             }
    #             QHeaderView::section {
    #                 background-color: #e6e6e6;
    #                 padding: 5px;
    #                 border: none;
    #                 font-weight: bold;
    #             }
    #         """)
    #         structure_table.setEditTriggers(QTableWidget.NoEditTriggers)
    #         structure_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    #         structure_table.verticalHeader().setVisible(False)
    #         structure_table.setSelectionMode(QTableWidget.NoSelection)
    #         structure_table.setMaximumHeight(120)
            
    #         structure_card_layout.addWidget(structure_table)
    #         summary_container_layout.addWidget(structure_card)
    #         summary_container_layout.addSpacing(10)
            
    #         # 두께 요약 카드
    #         thickness_card = QFrame()
    #         thickness_card.setFrameShape(QFrame.StyledPanel)
    #         thickness_card.setStyleSheet("QFrame { background-color: white; border: 1px solid #dddddd; border-radius: 5px; }")
    #         thickness_card_layout = QVBoxLayout(thickness_card)
            
    #         card_title = QLabel("Thickness Summary")
    #         card_title.setStyleSheet("font-size: 14pt; font-weight: bold; color: #003366;")
    #         thickness_card_layout.addWidget(card_title)
            
    #         # 계산된 두께 값 취합
    #         thickness_by_layer = {}
    #         for item in data['post_si_thickness_measurements'] + data['pre_si_thickness_measurements']:
    #             layer = item['layer']
    #             if layer not in thickness_by_layer:
    #                 thickness_by_layer[layer] = []
    #             thickness_by_layer[layer].append(item['thickness_nm'])
            
    #         # 두께 요약 테이블
    #         thickness_summary_table = QTableWidget()
    #         thickness_summary_table.setColumnCount(5)
    #         thickness_summary_table.setHorizontalHeaderLabels(["Layer", "Type", "Min (nm)", "Avg (nm)", "Max (nm)"])
    #         thickness_summary_table.setRowCount(len(thickness_by_layer))
            
    #         row = 0
    #         for layer_idx, thicknesses in thickness_by_layer.items():
    #             layer_type = "Si" if layer_idx in data['si_layers'] else "SiGe"
                
    #             thickness_summary_table.setItem(row, 0, QTableWidgetItem(f"Layer {layer_idx}"))
    #             thickness_summary_table.setItem(row, 1, QTableWidgetItem(layer_type))
    #             thickness_summary_table.setItem(row, 2, QTableWidgetItem(f"{min(thicknesses):.2f}"))
    #             thickness_summary_table.setItem(row, 3, QTableWidgetItem(f"{sum(thicknesses)/len(thicknesses):.2f}"))
    #             thickness_summary_table.setItem(row, 4, QTableWidgetItem(f"{max(thicknesses):.2f}"))
                
    #             # 행 색상 설정
    #             if layer_type == "Si":
    #                 for j in range(thickness_summary_table.columnCount()):
    #                     thickness_summary_table.item(row, j).setBackground(QColor(240, 248, 255))  # Alice Blue
    #             else:
    #                 for j in range(thickness_summary_table.columnCount()):
    #                     thickness_summary_table.item(row, j).setBackground(QColor(255, 240, 245))  # Lavender Blush
                        
    #             row += 1
            
    #         # 테이블 스타일
    #         thickness_summary_table.setStyleSheet("""
    #             QTableWidget {
    #                 gridline-color: #d4d4d4;
    #                 background-color: white;
    #                 border: none;
    #             }
    #             QHeaderView::section {
    #                 background-color: #e6e6e6;
    #                 padding: 5px;
    #                 border: none;
    #                 font-weight: bold;
    #             }
    #         """)
    #         thickness_summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
    #         thickness_summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    #         thickness_summary_table.verticalHeader().setVisible(False)
    #         thickness_summary_table.setMaximumHeight(25 * (len(thickness_by_layer) + 1) + 5)
            
    #         thickness_card_layout.addWidget(thickness_summary_table)
    #         summary_container_layout.addWidget(thickness_card)
            
    #         summary_scroll.setWidget(summary_container)
    #         summary_layout.addWidget(summary_scroll)
            
    #         tab_widget.addTab(summary_tab, "Summary")
            
    #         main_layout.addWidget(tab_widget)
            
    #         # 닫기 버튼
    #         button_container = QWidget()
    #         button_layout = QHBoxLayout(button_container)
    #         button_layout.setContentsMargins(0, 10, 0, 0)
            
    #         export_btn = QPushButton("Export Results")
    #         export_btn.setStyleSheet("""
    #             QPushButton {
    #                 background-color: #4CAF50;
    #                 color: white;
    #                 padding: 8px;
    #                 font-size: 12pt;
    #                 border-radius: 4px;
    #                 min-width: 150px;
    #             }
    #             QPushButton:hover {
    #                 background-color: #45a049;
    #             }
    #         """)
    #         button_layout.addWidget(export_btn)
            
    #         button_layout.addStretch()
            
    #         close_button = QPushButton("Close")
    #         close_button.setStyleSheet("""
    #             QPushButton {
    #                 background-color: #f2f2f2;
    #                 color: #333333;
    #                 padding: 8px;
    #                 font-size: 12pt;
    #                 border-radius: 4px;
    #                 min-width: 150px;
    #             }
    #             QPushButton:hover {
    #                 background-color: #e6e6e6;
    #             }
    #         """)
    #         button_layout.addWidget(close_button)
            
    #         main_layout.addWidget(button_container)
            
    #         # 버튼 이벤트 연결
    #         close_button.clicked.connect(result_dialog.accept)
    #         export_btn.clicked.connect(lambda: self.export_selectivity_results(data))
            
    #         result_dialog.setLayout(main_layout)
    #         result_dialog.exec_()
            
    #     except Exception as e:
    #         QMessageBox.critical(self, "Error", f"결과 표시 중 오류 발생: {str(e)}")
    #         traceback.print_exc()
                                        
    def show_selectivity_results(self, data):
        """선택비 계산 결과를 보여주는 향상된 다이얼로그"""
        if not hasattr(self, 'selectivity_data') or not self.selectivity_data:
            QMessageBox.warning(self, "No Data", "분석 결과가 없습니다.")
            return
            
        # 결과 다이얼로그 생성
        result_dialog = QDialog(self)
        result_dialog.setWindowTitle("Si/SiGe Selectivity Analysis Results")
        
        # 화면 크기에 따라 다이얼로그 크기 결정
        screen = QApplication.primaryScreen().geometry()
        dialog_width = min(int(screen.width() * 0.8), 1200)
        dialog_height = min(int(screen.height() * 0.8), 800)
        result_dialog.setMinimumSize(dialog_width, dialog_height)
        
        # 스타일시트 - 더 세련된 디자인
        result_dialog.setStyleSheet("""
            QDialog {
                background-color: #2D2D30;
                color: #E6E6E6;
            }
            QLabel {
                color: #E6E6E6;
                font-size: 12px;
            }
            QTabWidget {
                background-color: #2D2D30;
                color: #E6E6E6;
            }
            QTabWidget::pane {
                border: 1px solid #3E3E42;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #3E3E42;
                color: #E6E6E6;
                padding: 8px 20px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #007ACC;
            }
            QTabBar::tab:hover:!selected {
                background-color: #4A4A4D;
            }
            QTextEdit, QTableView, QTableWidget {
                background-color: #252526;
                color: #E6E6E6;
                border: 1px solid #3E3E42;
                border-radius: 2px;
                selection-background-color: #007ACC;
                font-size: 12px;
            }
            QPushButton {
                background-color: #0E639C;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1177BB;
            }
            QPushButton:pressed {
                background-color: #007ACC;
            }
            QHeaderView::section {
                background-color: #3E3E42;
                color: #E6E6E6;
                padding: 6px;
                border: 1px solid #2D2D30;
                font-weight: bold;
            }
            QTableWidget::item:selected {
                background-color: #264F78;
            }
            QComboBox {
                background-color: #3E3E42;
                color: #E6E6E6;
                border: 1px solid #3E3E42;
                border-radius: 3px;
                padding: 5px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2D2D30;
                color: #E6E6E6;
                selection-background-color: #264F78;
            }
        """)
        
        # 메인 레이아웃
        main_layout = QVBoxLayout(result_dialog)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # 헤더 타이틀 추가
        header_label = QLabel("Si/SiGe Selectivity Analysis Results")
        header_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #E6E6E6; margin-bottom: 10px;")
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)
        
        # 탭 위젯 생성
        tab_widget = QTabWidget()
        tab_widget.setDocumentMode(True)  # 더 간결한 탭 모양
        main_layout.addWidget(tab_widget)
        
        #------------------------
        # 요약 탭
        #------------------------
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        summary_layout.setContentsMargins(10, 15, 10, 10)
        summary_layout.setSpacing(15)
        
        # 요약 정보 섹션
        summary_info = QLabel("전체 분석 결과")
        summary_info.setStyleSheet("font-size: 14px; font-weight: bold; color: #E6E6E6;")
        summary_layout.addWidget(summary_info)
        
        # 요약 테이블
        summary_table = QTableWidget()
        summary_table.setColumnCount(6)
        summary_table.setHorizontalHeaderLabels([
            "SiGe Layer", "Si Layer", "SiGe Recess (nm)", 
            "Pre Si (nm)", "Post Si (nm)", "Si Loss (nm)"
        ])
        
        # 데이터 채우기
        sige_recess = []
        post_si = []
        pre_si = []
        si_loss = []
        sige_ids = []
        si_ids = []

        # 데이터 준비
        used_sige_indices = set()
        used_si_indices = set()
        sige_mapping = {}
        si_mapping = {}
        counter_sige = 1
        counter_si = 1

        for sige_idx, result in data['intersection_results'].items():
            si_idx = result['si_layer']
            sige_recess.append(result['SiGe_Recess'])

            if sige_idx not in sige_mapping:
                sige_mapping[sige_idx] = counter_sige
                counter_sige += 1
            
            # Si Layer 번호 매핑
            if si_idx not in si_mapping:
                si_mapping[si_idx] = counter_si
                counter_si += 1
            
            # 매핑된 번호 사용
            sige_ids.append(sige_mapping[sige_idx])
            si_ids.append(si_mapping[si_idx])
            
        for i in range(len(data['post_si_thickness_measurements'])):
            try:
                post_si.append(data['post_si_thickness_measurements'][i]['thickness_nm'])
            except:
                post_si.append(0)

        for i in range(len(data['pre_si_thickness_measurements'])):
            try:
                pre_si.append(data['pre_si_thickness_measurements'][i]['thickness_nm'])
            except:
                pre_si.append(0)

        # Si Loss 계산
        for i in range(min(len(pre_si), len(post_si))):
            si_loss.append(pre_si[i] - post_si[i])
        
        # 테이블 행 수 설정
        rows = len(sige_recess)
        summary_table.setRowCount(rows)
        
        # 컬러 스케일 설정
        def get_color_for_value(value, min_val=0, max_val=100):
            # 색상 그라데이션: 낮은 값 -> 빨간색, 높은 값 -> 초록색
            if value < min_val or value > max_val:
                return QColor(120, 120, 120, 80)  # 범위 외 값은 회색
                
            # 값을 0-1 범위로 정규화
            normalized = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
            
            # 빨간색 -> 노란색 -> 초록색 그라데이션
            if normalized < 0.5:
                # 빨간색 -> 노란색
                r = 220
                g = int(220 * normalized * 2)
                b = 60
            else:
                # 노란색 -> 초록색
                r = int(220 * (1 - (normalized - 0.5) * 2))
                g = 220
                b = 60
                
            return QColor(r, g, b, 80)
        
        # 데이터 채우기
        for row in range(rows):
            # SiGe Layer
            sige_layer_item = QTableWidgetItem(f"Layer {sige_ids[row]}")
            sige_layer_item.setTextAlignment(Qt.AlignCenter)
            summary_table.setItem(row, 0, sige_layer_item)
            
            # Si Layer
            si_layer_item = QTableWidgetItem(f"Layer {si_ids[row]}")
            si_layer_item.setTextAlignment(Qt.AlignCenter)
            summary_table.setItem(row, 1, si_layer_item)
            
            # SiGe Recess
            if row < len(sige_recess):
                recess_item = QTableWidgetItem(f"{sige_recess[row]:.2f}")
                recess_item.setTextAlignment(Qt.AlignCenter)
                recess_item.setBackground(get_color_for_value(sige_recess[row], 0, max(sige_recess) if sige_recess else 100))
                summary_table.setItem(row, 2, recess_item)
            
            # Pre Si
            if row < len(pre_si):
                pre_si_item = QTableWidgetItem(f"{pre_si[row]:.2f}")
                pre_si_item.setTextAlignment(Qt.AlignCenter)
                summary_table.setItem(row, 3, pre_si_item)
            
            # Post Si
            if row < len(post_si):
                post_si_item = QTableWidgetItem(f"{post_si[row]:.2f}")
                post_si_item.setTextAlignment(Qt.AlignCenter)
                summary_table.setItem(row, 4, post_si_item)
            
            # Si Loss
            if row < len(si_loss):
                si_loss_item = QTableWidgetItem(f"{si_loss[row]:.2f}")
                si_loss_item.setTextAlignment(Qt.AlignCenter)
                si_loss_item.setBackground(get_color_for_value(si_loss[row], 0, max(si_loss) if si_loss else 100))
                summary_table.setItem(row, 5, si_loss_item)
        
        # 평균 행 추가
        if rows > 0:
            summary_table.setRowCount(rows + 1)
            
            # 평균 계산
            avg_recess = sum(sige_recess) / len(sige_recess) if sige_recess else 0
            avg_pre_si = sum(pre_si) / len(pre_si) if pre_si else 0
            avg_post_si = sum(post_si) / len(post_si) if post_si else 0
            avg_si_loss = sum(si_loss) / len(si_loss) if si_loss else 0
            
            # 평균 행 설정
            avg_row = rows
            avg_label = QTableWidgetItem("평균")
            avg_label.setTextAlignment(Qt.AlignCenter)
            avg_label.setFont(QFont("Arial", weight=QFont.Bold))
            summary_table.setItem(avg_row, 0, avg_label)
            summary_table.setSpan(avg_row, 0, 1, 2)  # 첫 두 열 병합
            
            # SiGe Recess 평균
            recess_avg_item = QTableWidgetItem(f"{avg_recess:.2f}")
            recess_avg_item.setTextAlignment(Qt.AlignCenter)
            recess_avg_item.setFont(QFont("Arial", weight=QFont.Bold))
            recess_avg_item.setBackground(get_color_for_value(avg_recess, 0, max(sige_recess) if sige_recess else 100))
            summary_table.setItem(avg_row, 2, recess_avg_item)
            
            # Pre Si 평균
            pre_si_avg_item = QTableWidgetItem(f"{avg_pre_si:.2f}")
            pre_si_avg_item.setTextAlignment(Qt.AlignCenter)
            pre_si_avg_item.setFont(QFont("Arial", weight=QFont.Bold))
            summary_table.setItem(avg_row, 3, pre_si_avg_item)
            
            # Post Si 평균
            post_si_avg_item = QTableWidgetItem(f"{avg_post_si:.2f}")
            post_si_avg_item.setTextAlignment(Qt.AlignCenter)
            post_si_avg_item.setFont(QFont("Arial", weight=QFont.Bold))
            summary_table.setItem(avg_row, 4, post_si_avg_item)
            
            # Si Loss 평균
            si_loss_avg_item = QTableWidgetItem(f"{avg_si_loss:.2f}")
            si_loss_avg_item.setTextAlignment(Qt.AlignCenter)
            si_loss_avg_item.setFont(QFont("Arial", weight=QFont.Bold))
            si_loss_avg_item.setBackground(get_color_for_value(avg_si_loss, 0, max(si_loss) if si_loss else 100))
            summary_table.setItem(avg_row, 5, si_loss_avg_item)
            
            # 선택비 행 추가
            selectivity_row = rows + 1
            summary_table.setRowCount(rows + 2)
            
            selectivity_label = QTableWidgetItem("선택비")
            selectivity_label.setTextAlignment(Qt.AlignCenter)
            selectivity_label.setFont(QFont("Arial", weight=QFont.Bold))
            summary_table.setItem(selectivity_row, 0, selectivity_label)
            summary_table.setSpan(selectivity_row, 0, 1, 2)  # 첫 두 열 병합
            
            # 선택비 계산
            selectivity = avg_recess / avg_si_loss if avg_si_loss != 0 else float('inf')
            selectivity_value = QTableWidgetItem(f"{selectivity:.2f}")
            selectivity_value.setTextAlignment(Qt.AlignCenter)
            selectivity_value.setFont(QFont("Arial", weight=QFont.Bold))
            # 선택비 값에 대한 강조 색상 (높을수록 더 좋은 값)
            selectivity_value.setBackground(QColor(0, 220, 0, 100))  # 밝은 초록색
            selectivity_value.setForeground(QColor(255, 255, 255))   # 흰색 텍스트
            summary_table.setItem(selectivity_row, 2, selectivity_value)
            summary_table.setSpan(selectivity_row, 2, 1, 4)  # 나머지 열 병합
        
        # 테이블 설정
        summary_table.horizontalHeader().setStretchLastSection(True)
        summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        summary_table.verticalHeader().setVisible(False)
        summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        summary_table.setAlternatingRowColors(True)
        summary_table.setShowGrid(True)
        summary_layout.addWidget(summary_table)
        
        # 시각화 추가 (분석 이미지)
        if hasattr(data, 'plot') and data['plot']:
            visualization_group = QGroupBox("Selectivity Analysis Visualization")
            visualization_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 20px; margin-top: 20px; }")
            visualization_layout = QVBoxLayout(visualization_group)
            
            viz_image = QLabel()
            viz_image.setAlignment(Qt.AlignCenter)
            
            # 데이터에서 이미지 로드
            pixmap = QPixmap()
            pixmap.loadFromData(data['plot'].getvalue())
            
            # 이미지 크기 조정
            max_height = dialog_height - 300  # 대화 상자 크기에 맞게 최대 높이 설정
            scaled_pixmap = pixmap.scaled(
                dialog_width - 50, max_height, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            viz_image.setPixmap(scaled_pixmap)
            visualization_layout.addWidget(viz_image)
            summary_layout.addWidget(visualization_group)
        
        # 요약 탭 추가
        tab_widget.addTab(summary_tab, "결과 요약")
        
        #------------------------
        # 상세 측정 탭
        #------------------------
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)
        details_layout.setContentsMargins(10, 15, 10, 10)
        details_layout.setSpacing(15)
        
        # 측정 타입별 그룹화
        pre_si_group = QGroupBox("Pre Si Thickness Measurements")
        pre_si_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 20px; }")
        pre_si_layout = QVBoxLayout(pre_si_group)
        
        # Pre Si 테이블
        pre_si_table = QTableWidget()
        pre_si_table.setColumnCount(5)
        pre_si_table.setHorizontalHeaderLabels([
            "Point", "Layer", "Thickness (nm)", "Start Point", "End Point"
        ])
        
        # 데이터 채우기
        pre_si_measurements = data['pre_si_thickness_measurements']
        pre_si_table.setRowCount(len(pre_si_measurements))
        
        for row, measurement in enumerate(pre_si_measurements):
            # 포인트 번호
            point_item = QTableWidgetItem(f"P{row+1}")
            point_item.setTextAlignment(Qt.AlignCenter)
            pre_si_table.setItem(row, 0, point_item)
            
            # 레이어
            layer_item = QTableWidgetItem(f"Layer {measurement.get('layer', 'N/A')}")
            layer_item.setTextAlignment(Qt.AlignCenter)
            pre_si_table.setItem(row, 1, layer_item)
            
            # 두께
            thickness_item = QTableWidgetItem(f"{measurement.get('thickness_nm', 0):.2f}")
            thickness_item.setTextAlignment(Qt.AlignCenter)
            pre_si_table.setItem(row, 2, thickness_item)
            
            # 시작점
            if 'start_point' in measurement:
                start_x, start_y = measurement['start_point']
                start_item = QTableWidgetItem(f"({int(start_x)}, {int(start_y)})")
                start_item.setTextAlignment(Qt.AlignCenter)
                pre_si_table.setItem(row, 3, start_item)
            
            # 끝점
            if 'end_point' in measurement:
                end_x, end_y = measurement['end_point']
                end_item = QTableWidgetItem(f"({int(end_x)}, {int(end_y)})")
                end_item.setTextAlignment(Qt.AlignCenter)
                pre_si_table.setItem(row, 4, end_item)
        
        # 테이블 설정
        pre_si_table.horizontalHeader().setStretchLastSection(True)
        pre_si_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        pre_si_table.verticalHeader().setVisible(False)
        pre_si_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        pre_si_table.setAlternatingRowColors(True)
        pre_si_layout.addWidget(pre_si_table)
        details_layout.addWidget(pre_si_group)
        
        # Post Si 측정 그룹
        post_si_group = QGroupBox("Post Si Thickness Measurements")
        post_si_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 20px; }")
        post_si_layout = QVBoxLayout(post_si_group)
        
        # Post Si 테이블
        post_si_table = QTableWidget()
        post_si_table.setColumnCount(5)
        post_si_table.setHorizontalHeaderLabels([
            "Point", "Layer", "Thickness (nm)", "Start Point", "End Point"
        ])
        
        # 데이터 채우기
        post_si_measurements = data['post_si_thickness_measurements']
        post_si_table.setRowCount(len(post_si_measurements))
        
        for row, measurement in enumerate(post_si_measurements):
            # 포인트 번호
            point_item = QTableWidgetItem(f"P{row+1}")
            point_item.setTextAlignment(Qt.AlignCenter)
            post_si_table.setItem(row, 0, point_item)
            
            # 레이어
            layer_item = QTableWidgetItem(f"Layer {measurement.get('layer', 'N/A')}")
            layer_item.setTextAlignment(Qt.AlignCenter)
            post_si_table.setItem(row, 1, layer_item)
            
            # 두께
            thickness_item = QTableWidgetItem(f"{measurement.get('thickness_nm', 0):.2f}")
            thickness_item.setTextAlignment(Qt.AlignCenter)
            post_si_table.setItem(row, 2, thickness_item)
            
            # 시작점
            if 'start_point' in measurement:
                start_x, start_y = measurement['start_point']
                start_item = QTableWidgetItem(f"({int(start_x)}, {int(start_y)})")
                start_item.setTextAlignment(Qt.AlignCenter)
                post_si_table.setItem(row, 3, start_item)
            
            # 끝점
            if 'end_point' in measurement:
                end_x, end_y = measurement['end_point']
                end_item = QTableWidgetItem(f"({int(end_x)}, {int(end_y)})")
                end_item.setTextAlignment(Qt.AlignCenter)
                post_si_table.setItem(row, 4, end_item)
        
        # 테이블 설정
        post_si_table.horizontalHeader().setStretchLastSection(True)
        post_si_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        post_si_table.verticalHeader().setVisible(False)
        post_si_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        post_si_table.setAlternatingRowColors(True)
        post_si_layout.addWidget(post_si_table)
        details_layout.addWidget(post_si_group)
        
        # 상세 탭 추가
        tab_widget.addTab(details_tab, "상세 측정")
        
        #------------------------
        # 그래프 탭
        #------------------------
        graph_tab = QWidget()
        graph_layout = QVBoxLayout(graph_tab)
        graph_layout.setContentsMargins(10, 15, 10, 10)
        graph_layout.setSpacing(15)
        
        # 그래프 종류 선택
        graph_selector = QComboBox()
        graph_selector.addItems(["선택비 분석", "두께 분포", "레이어별 두께 비교"])
        graph_selector.setFixedWidth(200)
        graph_header = QHBoxLayout()
        graph_header.addWidget(QLabel("그래프 종류:"))
        graph_header.addWidget(graph_selector)
        graph_header.addStretch()
        graph_layout.addLayout(graph_header)
        
        # 그래프 스택 위젯
        graph_stack = QStackedWidget()
        graph_layout.addWidget(graph_stack)
        
        # 그래프 생성 (matplotlib)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        import matplotlib as mpl
        
        # 글꼴 및 스타일 설정
        mpl.rcParams['font.family'] = 'sans-serif'
        mpl.rcParams['font.size'] = 10
        mpl.rcParams['axes.linewidth'] = 0.8
        mpl.rcParams['axes.edgecolor'] = '#888888'
        
        #------------ 선택비 분석 그래프 ------------
        selectivity_graph = QWidget()
        selectivity_layout = QVBoxLayout(selectivity_graph)
        
        selectivity_fig = Figure(figsize=(8, 4), dpi=100, facecolor='#252526')
        selectivity_canvas = FigureCanvas(selectivity_fig)
        
        # 두 개의 서브플롯 생성 (왼쪽: 측정, 오른쪽: 선택비)
        gs = selectivity_fig.add_gridspec(1, 2, width_ratios=[1, 1])
        ax1 = selectivity_fig.add_subplot(gs[0, 0])
        ax2 = selectivity_fig.add_subplot(gs[0, 1])
        
        # 데이터 준비
        if len(sige_ids) > 0:
            x = range(len(sige_ids))
            width = 0.35
            
            # 첫 번째 그래프 (SiGe Recess와 Si Loss)
            if sige_recess and si_loss:
                bars1 = ax1.bar([p - width/2 for p in x], sige_recess, width, label='SiGe Recess', color='#3498db', alpha=0.8)
                bars2 = ax1.bar([p + width/2 for p in x], si_loss[:len(x)], width, label='Si Loss', color='#e74c3c', alpha=0.8)
                
                # 값 레이블 추가
                def add_labels(bars):
                    for bar in bars:
                        height = bar.get_height()
                        ax1.annotate(f'{height:.1f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 포인트 위
                                    textcoords="offset points",
                                    ha='center', va='bottom',
                                    fontsize=8, color='white')
                
                add_labels(bars1)
                add_labels(bars2)
                
                # 축 설정
                ax1.set_xlabel('Layer', fontsize=12, color='white', labelpad=10)
                ax1.set_ylabel('Thickness (nm)', fontsize=12, color='white', labelpad=10)
                ax1.set_title('SiGe Recess vs Si Loss', fontsize=13, color='white', pad=10)
                ax1.set_xticks(x)
                ax1.set_xticklabels([f'Layer {id}' for id in sige_ids])
                
                # 범례
                legend = ax1.legend(loc='upper right', fontsize=10, framealpha=0.7)
                for text in legend.get_texts():
                    text.set_color('white')
            
            # 두 번째 그래프 (선택비)
            if sige_recess and si_loss:
                # 각 레이어별 선택비 계산
                selectivities = []
                for i in range(min(len(sige_recess), len(si_loss))):
                    if si_loss[i] != 0:
                        selectivities.append(sige_recess[i] / si_loss[i])
                    else:
                        selectivities.append(0)
                
                bars3 = ax2.bar(x, selectivities, 0.5, color='#2ecc71', alpha=0.8)
                
                # 값 레이블 추가
                for bar in bars3:
                    height = bar.get_height()
                    ax2.annotate(f'{height:.1f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=8, color='white')
                
                # 평균 선택비 선 추가
                avg_selectivity = sum(selectivities) / len(selectivities) if selectivities else 0
                ax2.axhline(y=avg_selectivity, color='#f39c12', linestyle='--', linewidth=1.5, alpha=0.8)
                ax2.annotate(f'Avg: {avg_selectivity:.1f}',
                            xy=(len(x) - 1, avg_selectivity),
                            xytext=(5, 0),
                            textcoords="offset points",
                            ha='left', va='center',
                            fontsize=10, color='#f39c12')
                
                # 축 설정
                ax2.set_xlabel('Layer', fontsize=12, color='white', labelpad=10)
                ax2.set_ylabel('Selectivity Ratio', fontsize=12, color='white', labelpad=10)
                ax2.set_title('Si/SiGe Selectivity by Layer', fontsize=13, color='white', pad=10)
                ax2.set_xticks(x)
                ax2.set_xticklabels([f'Layer {id}' for id in sige_ids])
        else:
            # 데이터가 없는 경우
            ax1.text(0.5, 0.5, 'No measurement data available', 
                    ha='center', va='center', fontsize=12, color='white',
                    transform=ax1.transAxes)
            ax2.text(0.5, 0.5, 'No selectivity data available', 
                    ha='center', va='center', fontsize=12, color='white',
                    transform=ax2.transAxes)
        
        # 그래프 스타일 설정
        for ax in [ax1, ax2]:
            ax.set_facecolor('#252526')
            ax.tick_params(colors='white', grid_alpha=0.3)
            for spine in ax.spines.values():
                spine.set_color('#777777')
            ax.grid(True, linestyle='--', alpha=0.2, color='#777777')
        
        # 마진 조정
        selectivity_fig.tight_layout()
        selectivity_canvas.draw()
        
        selectivity_layout.addWidget(selectivity_canvas)
        graph_stack.addWidget(selectivity_graph)
        
        #------------ 두께 분포 그래프 ------------
        thickness_graph = QWidget()
        thickness_layout = QVBoxLayout(thickness_graph)
        
        thickness_fig = Figure(figsize=(8, 4), dpi=100, facecolor='#252526')
        thickness_canvas = FigureCanvas(thickness_fig)
        thickness_ax = thickness_fig.add_subplot(111)
        
        # 데이터 수집
        all_pre_thicknesses = [m['thickness_nm'] for m in pre_si_measurements] if pre_si_measurements else []
        all_post_thicknesses = [m['thickness_nm'] for m in post_si_measurements] if post_si_measurements else []
        
        if all_pre_thicknesses or all_post_thicknesses:
            # 박스플롯 데이터 준비
            data_to_plot = []
            labels = []
            
            if all_pre_thicknesses:
                data_to_plot.append(all_pre_thicknesses)
                labels.append('Pre Si')
            
            if all_post_thicknesses:
                data_to_plot.append(all_post_thicknesses)
                labels.append('Post Si')
            
            # 박스플롯 속성
            boxprops = dict(facecolor='#3498db', alpha=0.6, edgecolor='white', linewidth=1.5)
            whiskerprops = dict(color='white')
            capprops = dict(color='white')
            medianprops = dict(color='#f39c12', linewidth=1.5)
            flierprops = dict(marker='o', markerfacecolor='white', markersize=5, alpha=0.7)
            
            # 박스플롯 그리기
            thickness_ax.boxplot(
                data_to_plot,
                labels=labels,
                boxprops=boxprops,
                whiskerprops=whiskerprops,
                capprops=capprops,
                medianprops=medianprops,
                flierprops=flierprops,
                widths=0.5,
                patch_artist=True
            )
            
            # 각 포인트 산점도 추가 (지터 적용)
            for i, data in enumerate(data_to_plot):
                # 지터 계산 (포인트 간 겹침 방지)
                jitter = np.random.normal(0, 0.05, size=len(data))
                x = np.full_like(data, i+1) + jitter
                thickness_ax.scatter(x, data, alpha=0.6, color='#2ecc71', edgecolor='white', linewidth=0.5, s=30)
            
            # 평균값 표시
            for i, data in enumerate(data_to_plot):
                avg = sum(data) / len(data) if data else 0
                thickness_ax.plot([i+1-0.25, i+1+0.25], [avg, avg], 'r-', linewidth=2, color='#e74c3c')
                thickness_ax.text(i+1, avg*1.02, f'Avg: {avg:.2f}', ha='center', color='white', fontsize=9)
            
            # 축 설정
            thickness_ax.set_ylabel('Thickness (nm)', fontsize=12, color='white', labelpad=10)
            thickness_ax.set_title('Thickness Distribution', fontsize=14, color='white', pad=15)
        else:
            # 데이터가 없는 경우
            thickness_ax.text(0.5, 0.5, 'No thickness data available',
                            ha='center', va='center',
                            fontsize=14, color='white',
                            transform=thickness_ax.transAxes)
        
        # 그래프 스타일 설정
        thickness_ax.set_facecolor('#252526')
        thickness_ax.tick_params(colors='white', grid_alpha=0.3)
        for spine in thickness_ax.spines.values():
            spine.set_color('#777777')
        thickness_ax.grid(True, linestyle='--', alpha=0.2, color='#777777')
        
        # 마진 조정
        thickness_fig.tight_layout()
        thickness_canvas.draw()
        
        thickness_layout.addWidget(thickness_canvas)
        graph_stack.addWidget(thickness_graph)
        
        #------------ 레이어별 두께 비교 그래프 ------------
        layers_graph = QWidget()
        layers_layout = QVBoxLayout(layers_graph)
        
        layers_fig = Figure(figsize=(8, 4), dpi=100, facecolor='#252526')
        layers_canvas = FigureCanvas(layers_fig)
        layers_ax = layers_fig.add_subplot(111)
        
        # 레이어별 데이터 수집
        layer_data = {}
        
        # Pre Si 데이터 그룹화
        for m in pre_si_measurements:
            layer = m.get('layer', 'unknown')
            if layer not in layer_data:
                layer_data[layer] = {'pre': [], 'post': []}
            layer_data[layer]['pre'].append(m['thickness_nm'])
        
        # Post Si 데이터 그룹화
        for m in post_si_measurements:
            layer = m.get('layer', 'unknown')
            if layer not in layer_data:
                layer_data[layer] = {'pre': [], 'post': []}
            layer_data[layer]['post'].append(m['thickness_nm'])
        
        if layer_data:
            # 데이터 변환
            layers = sorted(layer_data.keys())
            x = np.arange(len(layers))
            width = 0.35
            
            # 각 레이어별 평균 계산
            pre_avgs = []
            post_avgs = []
            pre_stds = []
            post_stds = []
            
            for layer in layers:
                pre_data = layer_data[layer]['pre']
                post_data = layer_data[layer]['post']
                
                pre_avg = sum(pre_data) / len(pre_data) if pre_data else 0
                post_avg = sum(post_data) / len(post_data) if post_data else 0
                
                pre_std = np.std(pre_data) if len(pre_data) > 1 else 0
                post_std = np.std(post_data) if len(post_data) > 1 else 0
                
                pre_avgs.append(pre_avg)
                post_avgs.append(post_avg)
                pre_stds.append(pre_std)
                post_stds.append(post_std)
            
            # 그래프 그리기
            bars1 = layers_ax.bar(x - width/2, pre_avgs, width, label='Pre Si', 
                                color='#3498db', alpha=0.8, 
                                yerr=pre_stds, ecolor='white', capsize=5)
            
            bars2 = layers_ax.bar(x + width/2, post_avgs, width, label='Post Si', 
                                color='#e74c3c', alpha=0.8, 
                                yerr=post_stds, ecolor='white', capsize=5)
            
            # 값 레이블 추가
            def add_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    layers_ax.annotate(f'{height:.1f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom',
                                    fontsize=8, color='white')
            
            add_labels(bars1)
            add_labels(bars2)
            
            # 축 설정
            layers_ax.set_xlabel('Layer', fontsize=12, color='white', labelpad=10)
            layers_ax.set_ylabel('Thickness (nm)', fontsize=12, color='white', labelpad=10)
            layers_ax.set_title('Layer Thickness Comparison', fontsize=14, color='white', pad=15)
            layers_ax.set_xticks(x)
            layers_ax.set_xticklabels([f'Layer {layer}' for layer in layers])
            
            # 범례
            legend = layers_ax.legend(loc='upper right', fontsize=10, framealpha=0.7)
            for text in legend.get_texts():
                text.set_color('white')
                
            # Si Loss 표시
            for i, (pre, post) in enumerate(zip(pre_avgs, post_avgs)):
                if pre and post:
                    loss = pre - post
                    layers_ax.annotate(f'Loss: {loss:.1f}',
                                    xy=(i, min(pre, post)),
                                    xytext=(0, -15),
                                    textcoords="offset points",
                                    ha='center', va='top',
                                    fontsize=8, color='#f39c12')
                    
                    # 손실 화살표 표시
                    arrow_y = (pre + post) / 2
                    layers_ax.annotate('',
                                    xy=(i, post),
                                    xytext=(i, pre),
                                    arrowprops=dict(arrowstyle='<->', color='#f39c12', alpha=0.7))
            
        else:
            # 데이터가 없는 경우
            layers_ax.text(0.5, 0.5, 'No layer data available',
                        ha='center', va='center',
                        fontsize=14, color='white',
                        transform=layers_ax.transAxes)
        
        # 그래프 스타일 설정
        layers_ax.set_facecolor('#252526')
        layers_ax.tick_params(colors='white', grid_alpha=0.3)
        for spine in layers_ax.spines.values():
            spine.set_color('#777777')
        layers_ax.grid(True, linestyle='--', alpha=0.2, color='#777777')
        
        # 마진 조정
        layers_fig.tight_layout()
        layers_canvas.draw()
        
        layers_layout.addWidget(layers_canvas)
        graph_stack.addWidget(layers_graph)
        
        # 콤보박스 변경 시 스택 위젯 페이지 변경
        graph_selector.currentIndexChanged.connect(graph_stack.setCurrentIndex)
        
        # 그래프 탭 추가
        tab_widget.addTab(graph_tab, "그래프")
        
        #------------------------
        # 분석 결과 해석 탭
        #------------------------
        analysis_tab = QWidget()
        analysis_layout = QVBoxLayout(analysis_tab)
        analysis_layout.setContentsMargins(10, 15, 10, 10)
        analysis_layout.setSpacing(15)
        
        # 결과 해석 텍스트
        interpretation_text = QTextEdit()
        interpretation_text.setReadOnly(True)
        
        # 선택비 결과 분석
        avg_recess = sum(sige_recess) / len(sige_recess) if sige_recess else 0
        avg_pre_si = sum(pre_si) / len(pre_si) if pre_si else 0
        avg_post_si = sum(post_si) / len(post_si) if post_si else 0
        avg_si_loss = sum(si_loss) / len(si_loss) if si_loss else 0
        
        selectivity = avg_recess / avg_si_loss if avg_si_loss != 0 else float('inf')
        
        # 결과 해석 텍스트 작성
        interpretation_html = f"""
        <h2 style="color: #E6E6E6; text-align: center;">Si/SiGe 선택비 분석 결과</h2>
        
        <div style="margin: 20px; padding: 15px; background-color: #2A2A2A; border-radius: 5px;">
            <h3 style="color: #f39c12;">주요 결과</h3>
            <ul>
                <li>평균 SiGe 리세스: <b>{avg_recess:.2f} nm</b></li>
                <li>평균 Si 손실: <b>{avg_si_loss:.2f} nm</b></li>
                <li>선택비 (SiGe/Si): <b style="color: #2ecc71; font-size: 16px;">{selectivity:.2f}</b></li>
            </ul>
            
            <h3 style="color: #f39c12;">상세 정보</h3>
            <ul>
                <li>전체 측정 레이어 수: {len(sige_ids)}</li>
                <li>전처리 Si 평균 두께: {avg_pre_si:.2f} nm</li>
                <li>후처리 Si 평균 두께: {avg_post_si:.2f} nm</li>
            </ul>
        """
        
        interpretation_text.setHtml(interpretation_html)
        analysis_layout.addWidget(interpretation_text)
        
        # 분석 탭 추가
        tab_widget.addTab(analysis_tab, "분석 해석")
        
        #------------------------
        # 하단 버튼 영역
        #------------------------
        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 10, 0, 0)
        
        # 내보내기 버튼
        export_btn = QPushButton("결과 내보내기")
        export_btn.setFixedWidth(150)
        export_btn.setFixedHeight(35)
        export_btn.clicked.connect(lambda: self.export_selectivity_results(data))
        buttons_layout.addWidget(export_btn)
        
        # 공간 추가
        buttons_layout.addStretch()
        
        # 닫기 버튼
        close_btn = QPushButton("닫기")
        close_btn.setFixedWidth(100)
        close_btn.setFixedHeight(35)
        close_btn.clicked.connect(result_dialog.accept)
        buttons_layout.addWidget(close_btn)
        
        main_layout.addLayout(buttons_layout)
        
        # 다이얼로그 실행
        result_dialog.exec_()



class ScaleSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Scale Settings")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # 픽셀/nm 비율 설정
        ratio_group = QGroupBox("Pixel to nm ratio")
        ratio_layout = QFormLayout()
        self.ratio_input = QDoubleSpinBox()
        self.ratio_input.setRange(0.0001, 1000.0)
        self.ratio_input.setDecimals(4)
        self.ratio_input.setValue(self.parent.pixel_to_nm)
        ratio_layout.addRow("Ratio:", self.ratio_input)
        ratio_group.setLayout(ratio_layout)
        layout.addWidget(ratio_group)
        
        # 버튼
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
    def accept(self):
        self.parent.pixel_to_nm = self.ratio_input.value()
        # 측정값 업데이트
        if hasattr(self.parent, 'current_layer') and self.parent.current_layer is not None:
            current = self.parent.layers[self.parent.current_layer]
            if 'thickness_measurements' in current:
                # 기존 측정값들 업데이트
                for measurement in current['thickness_measurements']:
                    start = measurement['start_point']
                    end = measurement['end_point']
                    distance_pixels = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    measurement['distance'] = distance_pixels * self.parent.pixel_to_nm
        
        self.parent.display_image()  # 화면 갱신
        super().accept()


class AnalysisParametersDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Analysis Parameters")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        # SAM 모델 파라미터
        model_group = QGroupBox("SAM Model Parameters")
        model_layout = QFormLayout()
        
        self.confidence_threshold = QDoubleSpinBox()
        self.confidence_threshold.setRange(0.0, 1.0)
        self.confidence_threshold.setSingleStep(0.1)
        self.confidence_threshold.setValue(self.parent.confidence_threshold)
        model_layout.addRow("Confidence Threshold:", self.confidence_threshold)
        
        self.multimask_output = QCheckBox()
        self.multimask_output.setChecked(self.parent.multimask_output)
        model_layout.addRow("Multiple Masks Output:", self.multimask_output)
        model_layout.addRow("Multiple Masks Output:", self.multimask_output)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 측정 파라미터
        measurement_group = QGroupBox("Measurement Parameters")
        measurement_layout = QFormLayout()
        
        self.window_size = QSpinBox()
        self.window_size.setRange(10, 500)
        self.window_size.setValue(self.parent.window_size)
        measurement_layout.addRow("Window Size:", self.window_size)
                # 측정 포인트 개수 설정 추가
        self.measurement_points = QSpinBox()
        self.measurement_points.setRange(1, 10)  # 1~10개 포인트 허용
        self.measurement_points.setValue(self.parent.measurement_points)
        measurement_layout.addRow("Number of Measurement Points:", self.measurement_points)
        
        measurement_group.setLayout(measurement_layout)
        layout.addWidget(measurement_group)
        
        # 버튼
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)

    def accept(self):
        self.parent.confidence_threshold = self.confidence_threshold.value()
        self.parent.multimask_output = self.multimask_output.isChecked()
        self.parent.window_size = self.window_size.value()
        self.parent.measurement_points = self.measurement_points.value()
        
        # SAM 예측 함수 업데이트
        if self.parent.current_layer is not None:
            self.parent.update_mask()  # 마스크 재생성
        super().accept()

class AppearanceSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Appearance Settings")
        self.setModal(True)
        
        layout = QVBoxLayout()

            # 브러시 설정
        brush_group = QGroupBox("Brush Settings")
        brush_layout = QFormLayout()
        
        self.default_brush_size = QSpinBox()
        self.default_brush_size.setRange(1, 100)
        self.default_brush_size.setValue(self.parent.brush_size)
        brush_layout.addRow("Default Brush Size:", self.default_brush_size)
        
        brush_group.setLayout(brush_layout)
        layout.addWidget(brush_group)
        # 색상 설정
        color_group = QGroupBox("Color Settings")
        color_layout = QFormLayout()
        
        self.mask_opacity = QSlider(Qt.Horizontal)
        self.mask_opacity.setRange(0, 100)
        self.mask_opacity.setValue(self.parent.mask_opacity)
        color_layout.addRow("Mask Opacity:", self.mask_opacity)
        
        self.line_thickness = QSpinBox()
        self.line_thickness.setRange(1, 10)
        self.line_thickness.setValue(self.parent.line_thickness)
        color_layout.addRow("Line Thickness:", self.line_thickness)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        # Font 설정
        font_group = QGroupBox("Font Settings")
        font_layout = QFormLayout()
        
        self.font_size = QSpinBox()
        self.font_size.setRange(8, 72)
        self.font_size.setValue(self.parent.font_size)
        font_layout.addRow("Font Size:", self.font_size)
        
        font_group.setLayout(font_layout)
        layout.addWidget(font_group)
        
        # 버튼
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)

    def accept(self):
        self.parent.brush_size = self.default_brush_size.value()
        self.parent.brush_size_spinner.setValue(self.default_brush_size.value())
        self.parent.mask_opacity = self.mask_opacity.value()
        self.parent.line_thickness = self.line_thickness.value()
        self.parent.font_size = self.font_size.value()
        self.parent.display_image()  # 화면 갱신
        super().accept()

class ReportSettingsDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Replort Setting Parameters")
        self.setModal(True)
        layout = QVBoxLayout()
        
        # SAM 모델 파라미터
        model_group = QGroupBox("SAM Model Parameters")
        model_layout = QFormLayout()
        
        self.is_include_roughness = QCheckBox()

        # 버튼
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)

    def accept(self):
        self.parent.is_include_roughness = self.is_include_roughness.value()
        self.parent.display_image()  # 화면 갱신
        super().accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 스타일 설정
    app.setStyle('Fusion')
    
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = SAMApp()
    window.show()
    sys.exit(app.exec_())