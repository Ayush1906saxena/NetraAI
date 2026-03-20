import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../config/theme.dart';
import '../providers/screening_provider.dart';
import '../providers/connectivity_provider.dart';
import '../services/offline_queue.dart';

class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final screeningState = ref.watch(screeningProvider);
    final isOnline = ref.watch(isOnlineProvider);
    final pendingCount = OfflineQueue().length;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Netra AI'),
        actions: [
          // Sync status chip
          Padding(
            padding: const EdgeInsets.only(right: 12),
            child: _SyncChip(isOnline: isOnline, pendingCount: pendingCount),
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            // Hero banner
            _HeroBanner(),

            // New Screening CTA
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 16),
              child: ElevatedButton.icon(
                onPressed: () {
                  ref.read(screeningProvider.notifier).reset();
                  context.pushNamed('register');
                },
                icon: const Icon(Icons.add_circle_outline, size: 22),
                label: const Text('New Screening'),
              ),
            ),

            // Section header
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 24),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'Recent Screenings',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: NetraTheme.textPrimary,
                    ),
                  ),
                  TextButton(
                    onPressed: () => context.pushNamed('history'),
                    child: const Text('View All'),
                  ),
                ],
              ),
            ),

            // Recent screenings list
            Expanded(
              child: screeningState.recentScreenings.isEmpty
                  ? const _EmptyState()
                  : ListView.builder(
                      padding: const EdgeInsets.only(bottom: 24),
                      itemCount: screeningState.recentScreenings.length,
                      itemBuilder: (context, index) {
                        final s = screeningState.recentScreenings[index];
                        return Card(
                          child: ListTile(
                            leading: CircleAvatar(
                              backgroundColor:
                                  NetraTheme.primaryBlue.withOpacity(0.1),
                              child: const Icon(
                                Icons.person,
                                color: NetraTheme.primaryBlue,
                              ),
                            ),
                            title: Text(
                              s.patient.name,
                              style: const TextStyle(
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                            subtitle: Text(
                              '${_formatDate(s.createdAt)}  -  ${_statusLabel(s.status)}',
                              style: const TextStyle(fontSize: 13),
                            ),
                            trailing: _statusIcon(s.status),
                            onTap: () {
                              if (s.resultId != null) {
                                context.pushNamed('results',
                                    pathParameters: {'screeningId': s.id});
                              }
                            },
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: 0,
        onTap: (i) {
          if (i == 1) context.pushNamed('history');
        },
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
          BottomNavigationBarItem(icon: Icon(Icons.history), label: 'History'),
        ],
      ),
    );
  }

  String _formatDate(DateTime dt) {
    return '${dt.day}/${dt.month}/${dt.year}';
  }

  String _statusLabel(dynamic status) {
    return status.toString().split('.').last.replaceAll('_', ' ');
  }

  Widget _statusIcon(dynamic status) {
    final name = status.toString().split('.').last;
    switch (name) {
      case 'completed':
        return const Icon(Icons.check_circle, color: NetraTheme.riskLow);
      case 'analyzing':
        return const SizedBox(
          width: 20,
          height: 20,
          child: CircularProgressIndicator(strokeWidth: 2),
        );
      case 'failed':
        return const Icon(Icons.error, color: NetraTheme.riskHigh);
      default:
        return const Icon(Icons.circle_outlined, color: NetraTheme.textSecondary);
    }
  }
}

// ── Sub-widgets ────────────────────────────────────────────────────────

class _HeroBanner extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(24, 28, 24, 20),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          colors: [NetraTheme.primaryBlue, NetraTheme.primaryLight],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.only(
          bottomLeft: Radius.circular(24),
          bottomRight: Radius.circular(24),
        ),
      ),
      child: const Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.remove_red_eye_rounded, color: Colors.white70, size: 36),
          SizedBox(height: 10),
          Text(
            'AI-Powered\nDiabetic Retinopathy\nScreening',
            style: TextStyle(
              color: Colors.white,
              fontSize: 22,
              fontWeight: FontWeight.w700,
              height: 1.3,
            ),
          ),
          SizedBox(height: 6),
          Text(
            'Capture fundus images and get instant DR grading.',
            style: TextStyle(color: Colors.white70, fontSize: 14),
          ),
        ],
      ),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(Icons.camera_alt_outlined,
              size: 64, color: NetraTheme.textSecondary.withOpacity(0.4)),
          const SizedBox(height: 12),
          const Text(
            'No screenings yet',
            style: TextStyle(
              fontSize: 16,
              color: NetraTheme.textSecondary,
            ),
          ),
          const SizedBox(height: 4),
          const Text(
            'Tap "New Screening" to get started.',
            style: TextStyle(fontSize: 13, color: NetraTheme.textSecondary),
          ),
        ],
      ),
    );
  }
}

class _SyncChip extends StatelessWidget {
  final bool isOnline;
  final int pendingCount;

  const _SyncChip({required this.isOnline, required this.pendingCount});

  @override
  Widget build(BuildContext context) {
    return Chip(
      avatar: Icon(
        isOnline ? Icons.cloud_done : Icons.cloud_off,
        size: 16,
        color: isOnline ? NetraTheme.riskLow : NetraTheme.riskModerate,
      ),
      label: Text(
        isOnline
            ? (pendingCount > 0 ? '$pendingCount pending' : 'Online')
            : 'Offline ($pendingCount)',
        style: const TextStyle(fontSize: 12),
      ),
      backgroundColor: Colors.white,
      padding: EdgeInsets.zero,
      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
    );
  }
}
