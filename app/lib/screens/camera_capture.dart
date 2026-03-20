import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../config/theme.dart';
import '../models/screening.dart';
import '../providers/screening_provider.dart';
import '../services/camera_service.dart';
import '../widgets/eye_capture_widget.dart';
import '../widgets/iqa_feedback.dart';

class CameraCaptureScreen extends ConsumerStatefulWidget {
  final String screeningId;

  const CameraCaptureScreen({super.key, required this.screeningId});

  @override
  ConsumerState<CameraCaptureScreen> createState() =>
      _CameraCaptureScreenState();
}

class _CameraCaptureScreenState extends ConsumerState<CameraCaptureScreen> {
  final _camera = CameraService();

  Eye _selectedEye = Eye.left;
  CaptureResult? _leftCapture;
  CaptureResult? _rightCapture;
  bool _isCapturing = false;

  CaptureResult? get _currentCapture =>
      _selectedEye == Eye.left ? _leftCapture : _rightCapture;

  bool get _bothCaptured => _leftCapture != null && _rightCapture != null;

  Future<void> _captureImage() async {
    setState(() => _isCapturing = true);

    final result = await _camera.captureImage();

    if (result != null) {
      setState(() {
        if (_selectedEye == Eye.left) {
          _leftCapture = result;
        } else {
          _rightCapture = result;
        }
      });

      // Persist to screening state.
      await ref
          .read(screeningProvider.notifier)
          .addEyeImage(_selectedEye, result.filePath);
    }

    setState(() => _isCapturing = false);
  }

  Future<void> _retake() async {
    final path = _currentCapture?.filePath;
    if (path != null) await _camera.deleteCapture(path);

    setState(() {
      if (_selectedEye == Eye.left) {
        _leftCapture = null;
      } else {
        _rightCapture = null;
      }
    });
  }

  void _proceed() {
    context.pushNamed('processing',
        pathParameters: {'screeningId': widget.screeningId});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        title: const Text('Capture Fundus'),
        backgroundColor: Colors.black87,
      ),
      body: SafeArea(
        child: Column(
          children: [
            // Eye toggle
            _EyeToggle(
              selected: _selectedEye,
              leftCaptured: _leftCapture != null,
              rightCaptured: _rightCapture != null,
              onChanged: (eye) => setState(() => _selectedEye = eye),
            ),

            // Preview area
            Expanded(
              child: _currentCapture != null
                  ? _PreviewArea(
                      capture: _currentCapture!,
                      onRetake: _retake,
                    )
                  : EyeCaptureOverlay(
                      label:
                          _selectedEye == Eye.left ? 'Left Eye' : 'Right Eye',
                    ),
            ),

            // Quality feedback
            if (_currentCapture != null)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: IqaFeedbackWidget(quality: _currentCapture!.quality),
              ),

            const SizedBox(height: 16),

            // Action buttons
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 0, 24, 24),
              child: _currentCapture == null
                  ? _CaptureButton(
                      isCapturing: _isCapturing,
                      onPressed: _captureImage,
                    )
                  : _bothCaptured
                      ? ElevatedButton.icon(
                          onPressed: _proceed,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: NetraTheme.riskLow,
                          ),
                          icon: const Icon(Icons.arrow_forward),
                          label: const Text('Proceed to Analysis'),
                        )
                      : Row(
                          children: [
                            Expanded(
                              child: OutlinedButton(
                                onPressed: _retake,
                                style: OutlinedButton.styleFrom(
                                  foregroundColor: Colors.white,
                                  side: const BorderSide(color: Colors.white54),
                                ),
                                child: const Text('Retake'),
                              ),
                            ),
                            const SizedBox(width: 12),
                            Expanded(
                              child: ElevatedButton(
                                onPressed: () {
                                  setState(() {
                                    _selectedEye = _selectedEye == Eye.left
                                        ? Eye.right
                                        : Eye.left;
                                  });
                                },
                                child: Text(
                                  'Capture ${_selectedEye == Eye.left ? 'Right' : 'Left'} Eye',
                                ),
                              ),
                            ),
                          ],
                        ),
            ),
          ],
        ),
      ),
    );
  }
}

// ── Sub-widgets ────────────────────────────────────────────────────────

class _EyeToggle extends StatelessWidget {
  final Eye selected;
  final bool leftCaptured;
  final bool rightCaptured;
  final ValueChanged<Eye> onChanged;

  const _EyeToggle({
    required this.selected,
    required this.leftCaptured,
    required this.rightCaptured,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white12,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          _tab('Left Eye', Eye.left, leftCaptured),
          _tab('Right Eye', Eye.right, rightCaptured),
        ],
      ),
    );
  }

  Widget _tab(String label, Eye eye, bool captured) {
    final isActive = selected == eye;
    return Expanded(
      child: GestureDetector(
        onTap: () => onChanged(eye),
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 250),
          padding: const EdgeInsets.symmetric(vertical: 12),
          decoration: BoxDecoration(
            color: isActive ? NetraTheme.primaryBlue : Colors.transparent,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              if (captured)
                const Padding(
                  padding: EdgeInsets.only(right: 6),
                  child:
                      Icon(Icons.check_circle, color: NetraTheme.riskLow, size: 18),
                ),
              Text(
                label,
                textAlign: TextAlign.center,
                style: TextStyle(
                  color: isActive ? Colors.white : Colors.white60,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _PreviewArea extends StatelessWidget {
  final CaptureResult capture;
  final VoidCallback onRetake;

  const _PreviewArea({required this.capture, required this.onRetake});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(16),
        child: Image.file(
          File(capture.filePath),
          fit: BoxFit.contain,
        ),
      ),
    );
  }
}

class _CaptureButton extends StatelessWidget {
  final bool isCapturing;
  final VoidCallback onPressed;

  const _CaptureButton({required this.isCapturing, required this.onPressed});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: isCapturing ? null : onPressed,
      child: Container(
        width: 72,
        height: 72,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          color: Colors.white,
          border: Border.all(color: NetraTheme.primaryBlue, width: 4),
          boxShadow: [
            BoxShadow(
              color: NetraTheme.primaryBlue.withOpacity(0.4),
              blurRadius: 16,
              spreadRadius: 2,
            ),
          ],
        ),
        child: isCapturing
            ? const Padding(
                padding: EdgeInsets.all(18),
                child: CircularProgressIndicator(strokeWidth: 3),
              )
            : const Icon(
                Icons.camera_alt,
                color: NetraTheme.primaryBlue,
                size: 32,
              ),
      ),
    );
  }
}
