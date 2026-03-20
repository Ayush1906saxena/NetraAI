import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../config/theme.dart';
import '../providers/screening_provider.dart';

class ProcessingScreen extends ConsumerStatefulWidget {
  final String screeningId;

  const ProcessingScreen({super.key, required this.screeningId});

  @override
  ConsumerState<ProcessingScreen> createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends ConsumerState<ProcessingScreen>
    with SingleTickerProviderStateMixin {
  late final AnimationController _pulseCtrl;
  late final Animation<double> _pulse;

  String _statusMessage = 'Uploading images...';
  bool _hasError = false;

  @override
  void initState() {
    super.initState();

    _pulseCtrl = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat(reverse: true);

    _pulse = Tween<double>(begin: 0.85, end: 1.0).animate(
      CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut),
    );

    _startAnalysis();
  }

  Future<void> _startAnalysis() async {
    try {
      setState(() => _statusMessage = 'Uploading images...');

      // Wait briefly for upload progress (provider manages actual upload).
      await Future.delayed(const Duration(seconds: 1));
      if (!mounted) return;

      setState(() => _statusMessage = 'Analyzing fundus images...');
      await ref.read(screeningProvider.notifier).triggerAnalysis();

      if (!mounted) return;

      final state = ref.read(screeningProvider);
      if (state.error != null) {
        setState(() {
          _hasError = true;
          _statusMessage = state.error!;
        });
      } else {
        // Navigate to results.
        context.goNamed('results',
            pathParameters: {'screeningId': widget.screeningId});
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _hasError = true;
          _statusMessage = 'Something went wrong. Please try again.';
        });
      }
    }
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final uploadProgress = ref.watch(screeningProvider).uploadProgress;

    return Scaffold(
      backgroundColor: NetraTheme.surfaceWhite,
      body: SafeArea(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 40),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Animated eye icon
                ScaleTransition(
                  scale: _pulse,
                  child: Container(
                    width: 120,
                    height: 120,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: _hasError
                          ? NetraTheme.riskHigh.withOpacity(0.1)
                          : NetraTheme.primaryBlue.withOpacity(0.1),
                    ),
                    child: Icon(
                      _hasError
                          ? Icons.error_outline
                          : Icons.remove_red_eye_rounded,
                      size: 56,
                      color: _hasError
                          ? NetraTheme.riskHigh
                          : NetraTheme.primaryBlue,
                    ),
                  ),
                ),
                const SizedBox(height: 32),

                // Status text
                Text(
                  _hasError ? 'Error' : 'Processing',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.w700,
                    color: _hasError
                        ? NetraTheme.riskHigh
                        : NetraTheme.textPrimary,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  _statusMessage,
                  textAlign: TextAlign.center,
                  style: const TextStyle(
                    fontSize: 15,
                    color: NetraTheme.textSecondary,
                  ),
                ),
                const SizedBox(height: 32),

                // Progress indicator
                if (!_hasError) ...[
                  ClipRRect(
                    borderRadius: BorderRadius.circular(6),
                    child: LinearProgressIndicator(
                      value: uploadProgress < 1.0 ? uploadProgress : null,
                      backgroundColor:
                          NetraTheme.primaryBlue.withOpacity(0.12),
                      valueColor: const AlwaysStoppedAnimation<Color>(
                        NetraTheme.primaryBlue,
                      ),
                      minHeight: 6,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    uploadProgress < 1.0
                        ? 'Uploading: ${(uploadProgress * 100).toStringAsFixed(0)}%'
                        : 'AI analysis in progress...',
                    style: const TextStyle(
                      fontSize: 13,
                      color: NetraTheme.textSecondary,
                    ),
                  ),
                ],

                // Retry button on error
                if (_hasError) ...[
                  const SizedBox(height: 24),
                  ElevatedButton.icon(
                    onPressed: () {
                      setState(() {
                        _hasError = false;
                        _statusMessage = 'Retrying...';
                      });
                      _startAnalysis();
                    },
                    icon: const Icon(Icons.refresh),
                    label: const Text('Retry'),
                  ),
                  const SizedBox(height: 12),
                  TextButton(
                    onPressed: () => context.go('/'),
                    child: const Text('Back to Home'),
                  ),
                ],
              ],
            ),
          ),
        ),
      ),
    );
  }
}
