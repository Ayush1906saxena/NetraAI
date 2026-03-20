import 'package:json_annotation/json_annotation.dart';
import 'patient.dart';

part 'screening.g.dart';

/// Overall status of a screening session.
enum ScreeningStatus {
  @JsonValue('created')
  created,
  @JsonValue('capturing')
  capturing,
  @JsonValue('uploading')
  uploading,
  @JsonValue('analyzing')
  analyzing,
  @JsonValue('completed')
  completed,
  @JsonValue('failed')
  failed,
  @JsonValue('queued_offline')
  queuedOffline,
}

/// Which eye an image belongs to.
enum Eye {
  @JsonValue('left')
  left,
  @JsonValue('right')
  right,
}

@JsonSerializable()
class EyeImage {
  final String? id;
  final Eye eye;
  final String localPath;
  final String? remoteUrl;
  final bool uploaded;
  final String? qualityScore;

  const EyeImage({
    this.id,
    required this.eye,
    required this.localPath,
    this.remoteUrl,
    this.uploaded = false,
    this.qualityScore,
  });

  factory EyeImage.fromJson(Map<String, dynamic> json) =>
      _$EyeImageFromJson(json);
  Map<String, dynamic> toJson() => _$EyeImageToJson(this);

  EyeImage copyWith({
    String? id,
    Eye? eye,
    String? localPath,
    String? remoteUrl,
    bool? uploaded,
    String? qualityScore,
  }) {
    return EyeImage(
      id: id ?? this.id,
      eye: eye ?? this.eye,
      localPath: localPath ?? this.localPath,
      remoteUrl: remoteUrl ?? this.remoteUrl,
      uploaded: uploaded ?? this.uploaded,
      qualityScore: qualityScore ?? this.qualityScore,
    );
  }
}

@JsonSerializable()
class Screening {
  final String id;
  final Patient patient;
  final ScreeningStatus status;
  final List<EyeImage> images;
  final String? resultId;
  final DateTime createdAt;
  final DateTime? completedAt;

  const Screening({
    required this.id,
    required this.patient,
    this.status = ScreeningStatus.created,
    this.images = const [],
    this.resultId,
    required this.createdAt,
    this.completedAt,
  });

  factory Screening.fromJson(Map<String, dynamic> json) =>
      _$ScreeningFromJson(json);
  Map<String, dynamic> toJson() => _$ScreeningToJson(this);

  Screening copyWith({
    String? id,
    Patient? patient,
    ScreeningStatus? status,
    List<EyeImage>? images,
    String? resultId,
    DateTime? createdAt,
    DateTime? completedAt,
  }) {
    return Screening(
      id: id ?? this.id,
      patient: patient ?? this.patient,
      status: status ?? this.status,
      images: images ?? this.images,
      resultId: resultId ?? this.resultId,
      createdAt: createdAt ?? this.createdAt,
      completedAt: completedAt ?? this.completedAt,
    );
  }

  /// Whether both eyes have been captured.
  bool get bothEyesCaptured =>
      images.any((i) => i.eye == Eye.left) &&
      images.any((i) => i.eye == Eye.right);
}

// ── Manual serialisation ───────────────────────────────────────────────

EyeImage _$EyeImageFromJson(Map<String, dynamic> json) => EyeImage(
      id: json['id'] as String?,
      eye: $enumDecode(_$EyeEnumMap, json['eye']),
      localPath: json['local_path'] as String,
      remoteUrl: json['remote_url'] as String?,
      uploaded: json['uploaded'] as bool? ?? false,
      qualityScore: json['quality_score'] as String?,
    );

Map<String, dynamic> _$EyeImageToJson(EyeImage instance) => <String, dynamic>{
      'id': instance.id,
      'eye': _$EyeEnumMap[instance.eye]!,
      'local_path': instance.localPath,
      'remote_url': instance.remoteUrl,
      'uploaded': instance.uploaded,
      'quality_score': instance.qualityScore,
    };

const _$EyeEnumMap = {
  Eye.left: 'left',
  Eye.right: 'right',
};

Screening _$ScreeningFromJson(Map<String, dynamic> json) => Screening(
      id: json['id'] as String,
      patient: Patient.fromJson(json['patient'] as Map<String, dynamic>),
      status: $enumDecode(_$ScreeningStatusEnumMap, json['status']),
      images: (json['images'] as List<dynamic>?)
              ?.map((e) => EyeImage.fromJson(e as Map<String, dynamic>))
              .toList() ??
          [],
      resultId: json['result_id'] as String?,
      createdAt: DateTime.parse(json['created_at'] as String),
      completedAt: json['completed_at'] == null
          ? null
          : DateTime.parse(json['completed_at'] as String),
    );

Map<String, dynamic> _$ScreeningToJson(Screening instance) =>
    <String, dynamic>{
      'id': instance.id,
      'patient': instance.patient.toJson(),
      'status': _$ScreeningStatusEnumMap[instance.status]!,
      'images': instance.images.map((e) => e.toJson()).toList(),
      'result_id': instance.resultId,
      'created_at': instance.createdAt.toIso8601String(),
      'completed_at': instance.completedAt?.toIso8601String(),
    };

const _$ScreeningStatusEnumMap = {
  ScreeningStatus.created: 'created',
  ScreeningStatus.capturing: 'capturing',
  ScreeningStatus.uploading: 'uploading',
  ScreeningStatus.analyzing: 'analyzing',
  ScreeningStatus.completed: 'completed',
  ScreeningStatus.failed: 'failed',
  ScreeningStatus.queuedOffline: 'queued_offline',
};
