import 'package:json_annotation/json_annotation.dart';

part 'patient.g.dart';

/// Gender options for patient registration.
enum Gender {
  @JsonValue('male')
  male,
  @JsonValue('female')
  female,
  @JsonValue('other')
  other,
}

/// Diabetes status reported by the patient.
enum DiabetesStatus {
  @JsonValue('type1')
  type1,
  @JsonValue('type2')
  type2,
  @JsonValue('gestational')
  gestational,
  @JsonValue('none')
  none,
  @JsonValue('unknown')
  unknown,
}

@JsonSerializable()
class Patient {
  final String? id;
  final String name;
  final int age;
  final Gender gender;
  final String phone;
  final DiabetesStatus diabetesStatus;
  final bool consentGiven;
  final DateTime? createdAt;

  const Patient({
    this.id,
    required this.name,
    required this.age,
    required this.gender,
    required this.phone,
    required this.diabetesStatus,
    required this.consentGiven,
    this.createdAt,
  });

  factory Patient.fromJson(Map<String, dynamic> json) =>
      _$PatientFromJson(json);

  Map<String, dynamic> toJson() => _$PatientToJson(this);

  Patient copyWith({
    String? id,
    String? name,
    int? age,
    Gender? gender,
    String? phone,
    DiabetesStatus? diabetesStatus,
    bool? consentGiven,
    DateTime? createdAt,
  }) {
    return Patient(
      id: id ?? this.id,
      name: name ?? this.name,
      age: age ?? this.age,
      gender: gender ?? this.gender,
      phone: phone ?? this.phone,
      diabetesStatus: diabetesStatus ?? this.diabetesStatus,
      consentGiven: consentGiven ?? this.consentGiven,
      createdAt: createdAt ?? this.createdAt,
    );
  }
}

// ── Manual serialisation fallback (used until build_runner is run) ──────
// The generated file patient.g.dart will override these when available.
Patient _$PatientFromJson(Map<String, dynamic> json) => Patient(
      id: json['id'] as String?,
      name: json['name'] as String,
      age: (json['age'] as num).toInt(),
      gender: $enumDecode(_$GenderEnumMap, json['gender']),
      phone: json['phone'] as String,
      diabetesStatus:
          $enumDecode(_$DiabetesStatusEnumMap, json['diabetes_status']),
      consentGiven: json['consent_given'] as bool,
      createdAt: json['created_at'] == null
          ? null
          : DateTime.parse(json['created_at'] as String),
    );

Map<String, dynamic> _$PatientToJson(Patient instance) => <String, dynamic>{
      'id': instance.id,
      'name': instance.name,
      'age': instance.age,
      'gender': _$GenderEnumMap[instance.gender]!,
      'phone': instance.phone,
      'diabetes_status': _$DiabetesStatusEnumMap[instance.diabetesStatus]!,
      'consent_given': instance.consentGiven,
      'created_at': instance.createdAt?.toIso8601String(),
    };

const _$GenderEnumMap = {
  Gender.male: 'male',
  Gender.female: 'female',
  Gender.other: 'other',
};

const _$DiabetesStatusEnumMap = {
  DiabetesStatus.type1: 'type1',
  DiabetesStatus.type2: 'type2',
  DiabetesStatus.gestational: 'gestational',
  DiabetesStatus.none: 'none',
  DiabetesStatus.unknown: 'unknown',
};
