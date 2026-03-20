import 'package:json_annotation/json_annotation.dart';

part 'report.g.dart';

/// Risk level determined by analysis — drives the traffic-light display.
enum RiskLevel {
  @JsonValue('low')
  low,
  @JsonValue('moderate')
  moderate,
  @JsonValue('high')
  high,
}

@JsonSerializable()
class EyeResult {
  final String eye; // 'left' or 'right'
  final int drGrade; // 0-4
  final String drLabel;
  final double confidence;
  final String? gradcamUrl;
  final RiskLevel riskLevel;

  const EyeResult({
    required this.eye,
    required this.drGrade,
    required this.drLabel,
    required this.confidence,
    this.gradcamUrl,
    required this.riskLevel,
  });

  factory EyeResult.fromJson(Map<String, dynamic> json) =>
      _$EyeResultFromJson(json);
  Map<String, dynamic> toJson() => _$EyeResultToJson(this);
}

@JsonSerializable()
class Report {
  final String id;
  final String screeningId;
  final List<EyeResult> eyeResults;
  final RiskLevel overallRisk;
  final String recommendation;
  final bool referralNeeded;
  final String? referralUrgency;
  final String? reportPdfUrl;
  final DateTime createdAt;

  const Report({
    required this.id,
    required this.screeningId,
    required this.eyeResults,
    required this.overallRisk,
    required this.recommendation,
    required this.referralNeeded,
    this.referralUrgency,
    this.reportPdfUrl,
    required this.createdAt,
  });

  factory Report.fromJson(Map<String, dynamic> json) =>
      _$ReportFromJson(json);
  Map<String, dynamic> toJson() => _$ReportToJson(this);

  /// Convenience: worst-case DR grade across both eyes.
  int get maxDrGrade =>
      eyeResults.fold(0, (max, r) => r.drGrade > max ? r.drGrade : max);
}

// ── Manual serialisation ───────────────────────────────────────────────

EyeResult _$EyeResultFromJson(Map<String, dynamic> json) => EyeResult(
      eye: json['eye'] as String,
      drGrade: (json['dr_grade'] as num).toInt(),
      drLabel: json['dr_label'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      gradcamUrl: json['gradcam_url'] as String?,
      riskLevel: $enumDecode(_$RiskLevelEnumMap, json['risk_level']),
    );

Map<String, dynamic> _$EyeResultToJson(EyeResult instance) =>
    <String, dynamic>{
      'eye': instance.eye,
      'dr_grade': instance.drGrade,
      'dr_label': instance.drLabel,
      'confidence': instance.confidence,
      'gradcam_url': instance.gradcamUrl,
      'risk_level': _$RiskLevelEnumMap[instance.riskLevel]!,
    };

Report _$ReportFromJson(Map<String, dynamic> json) => Report(
      id: json['id'] as String,
      screeningId: json['screening_id'] as String,
      eyeResults: (json['eye_results'] as List<dynamic>)
          .map((e) => EyeResult.fromJson(e as Map<String, dynamic>))
          .toList(),
      overallRisk: $enumDecode(_$RiskLevelEnumMap, json['overall_risk']),
      recommendation: json['recommendation'] as String,
      referralNeeded: json['referral_needed'] as bool,
      referralUrgency: json['referral_urgency'] as String?,
      reportPdfUrl: json['report_pdf_url'] as String?,
      createdAt: DateTime.parse(json['created_at'] as String),
    );

Map<String, dynamic> _$ReportToJson(Report instance) => <String, dynamic>{
      'id': instance.id,
      'screening_id': instance.screeningId,
      'eye_results': instance.eyeResults.map((e) => e.toJson()).toList(),
      'overall_risk': _$RiskLevelEnumMap[instance.overallRisk]!,
      'recommendation': instance.recommendation,
      'referral_needed': instance.referralNeeded,
      'referral_urgency': instance.referralUrgency,
      'report_pdf_url': instance.reportPdfUrl,
      'created_at': instance.createdAt.toIso8601String(),
    };

const _$RiskLevelEnumMap = {
  RiskLevel.low: 'low',
  RiskLevel.moderate: 'moderate',
  RiskLevel.high: 'high',
};
