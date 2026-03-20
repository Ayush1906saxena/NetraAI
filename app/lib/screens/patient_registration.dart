import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../config/theme.dart';
import '../models/patient.dart';
import '../providers/screening_provider.dart';

class PatientRegistrationScreen extends ConsumerStatefulWidget {
  const PatientRegistrationScreen({super.key});

  @override
  ConsumerState<PatientRegistrationScreen> createState() =>
      _PatientRegistrationScreenState();
}

class _PatientRegistrationScreenState
    extends ConsumerState<PatientRegistrationScreen> {
  final _formKey = GlobalKey<FormState>();

  final _nameCtrl = TextEditingController();
  final _ageCtrl = TextEditingController();
  final _phoneCtrl = TextEditingController();

  Gender _gender = Gender.male;
  DiabetesStatus _diabetes = DiabetesStatus.unknown;
  bool _consentGiven = false;
  bool _submitting = false;

  @override
  void dispose() {
    _nameCtrl.dispose();
    _ageCtrl.dispose();
    _phoneCtrl.dispose();
    super.dispose();
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;

    if (!_consentGiven) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Patient consent is required to proceed.'),
          backgroundColor: NetraTheme.riskHigh,
        ),
      );
      return;
    }

    setState(() => _submitting = true);

    final patient = Patient(
      name: _nameCtrl.text.trim(),
      age: int.parse(_ageCtrl.text.trim()),
      gender: _gender,
      phone: _phoneCtrl.text.trim(),
      diabetesStatus: _diabetes,
      consentGiven: _consentGiven,
    );

    try {
      final screeningId =
          await ref.read(screeningProvider.notifier).createScreening(patient);
      if (mounted) {
        context.pushNamed('capture',
            pathParameters: {'screeningId': screeningId});
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Error: $e'),
            backgroundColor: NetraTheme.riskHigh,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Patient Registration')),
      body: SafeArea(
        child: Form(
          key: _formKey,
          child: ListView(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
            children: [
              const Text(
                'Enter Patient Details',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.w700,
                  color: NetraTheme.textPrimary,
                ),
              ),
              const SizedBox(height: 6),
              const Text(
                'All fields are required for accurate screening.',
                style: TextStyle(fontSize: 14, color: NetraTheme.textSecondary),
              ),
              const SizedBox(height: 24),

              // Name
              TextFormField(
                controller: _nameCtrl,
                decoration: const InputDecoration(
                  labelText: 'Full Name',
                  prefixIcon: Icon(Icons.person_outline),
                ),
                textCapitalization: TextCapitalization.words,
                validator: (v) =>
                    (v == null || v.trim().isEmpty) ? 'Name is required' : null,
              ),
              const SizedBox(height: 16),

              // Age
              TextFormField(
                controller: _ageCtrl,
                decoration: const InputDecoration(
                  labelText: 'Age',
                  prefixIcon: Icon(Icons.cake_outlined),
                ),
                keyboardType: TextInputType.number,
                inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                validator: (v) {
                  if (v == null || v.trim().isEmpty) return 'Age is required';
                  final age = int.tryParse(v.trim());
                  if (age == null || age < 1 || age > 120) {
                    return 'Enter a valid age';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 16),

              // Gender
              _SectionLabel('Gender'),
              const SizedBox(height: 8),
              SegmentedButton<Gender>(
                segments: const [
                  ButtonSegment(value: Gender.male, label: Text('Male')),
                  ButtonSegment(value: Gender.female, label: Text('Female')),
                  ButtonSegment(value: Gender.other, label: Text('Other')),
                ],
                selected: {_gender},
                onSelectionChanged: (v) => setState(() => _gender = v.first),
                style: SegmentedButton.styleFrom(
                  selectedBackgroundColor:
                      NetraTheme.primaryBlue.withOpacity(0.12),
                  selectedForegroundColor: NetraTheme.primaryBlue,
                ),
              ),
              const SizedBox(height: 16),

              // Phone
              TextFormField(
                controller: _phoneCtrl,
                decoration: const InputDecoration(
                  labelText: 'Phone Number',
                  prefixIcon: Icon(Icons.phone_outlined),
                ),
                keyboardType: TextInputType.phone,
                inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                validator: (v) {
                  if (v == null || v.trim().isEmpty) {
                    return 'Phone number is required';
                  }
                  if (v.trim().length < 10) return 'Enter a valid phone number';
                  return null;
                },
              ),
              const SizedBox(height: 16),

              // Diabetes status
              _SectionLabel('Diabetes Status'),
              const SizedBox(height: 8),
              DropdownButtonFormField<DiabetesStatus>(
                value: _diabetes,
                decoration: const InputDecoration(
                  prefixIcon: Icon(Icons.medical_information_outlined),
                ),
                items: const [
                  DropdownMenuItem(
                    value: DiabetesStatus.type1,
                    child: Text('Type 1 Diabetes'),
                  ),
                  DropdownMenuItem(
                    value: DiabetesStatus.type2,
                    child: Text('Type 2 Diabetes'),
                  ),
                  DropdownMenuItem(
                    value: DiabetesStatus.gestational,
                    child: Text('Gestational Diabetes'),
                  ),
                  DropdownMenuItem(
                    value: DiabetesStatus.none,
                    child: Text('No Diabetes'),
                  ),
                  DropdownMenuItem(
                    value: DiabetesStatus.unknown,
                    child: Text('Unknown'),
                  ),
                ],
                onChanged: (v) {
                  if (v != null) setState(() => _diabetes = v);
                },
              ),
              const SizedBox(height: 20),

              // Consent
              CheckboxListTile(
                value: _consentGiven,
                onChanged: (v) =>
                    setState(() => _consentGiven = v ?? false),
                controlAffinity: ListTileControlAffinity.leading,
                activeColor: NetraTheme.primaryBlue,
                title: const Text(
                  'Patient has given informed consent for fundus '
                  'imaging and AI-assisted DR screening.',
                  style: TextStyle(fontSize: 14),
                ),
                contentPadding: EdgeInsets.zero,
              ),
              const SizedBox(height: 28),

              // Submit
              ElevatedButton(
                onPressed: _submitting ? null : _submit,
                child: _submitting
                    ? const SizedBox(
                        width: 22,
                        height: 22,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Text('Continue to Capture'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  final String text;
  const _SectionLabel(this.text);

  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      style: const TextStyle(
        fontSize: 14,
        fontWeight: FontWeight.w600,
        color: NetraTheme.textPrimary,
      ),
    );
  }
}
