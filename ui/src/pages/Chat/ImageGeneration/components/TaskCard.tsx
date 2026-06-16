import { useEffect, useState, useRef, type ReactNode } from 'react';
import type { TaskRecord } from '../types';
import {
  useStore,
  ensureImageThumbnailCached,
  subscribeImageThumbnail,
  retryTask,
} from '../store';
import { formatImageRatio } from '../lib/size';
import { isAgentTaskPromptPending } from '../lib/taskPromptDisplay';
import ViewportTooltip from './ViewportTooltip';

interface Props {
  task: TaskRecord;
  onReuse: () => void;
  onEditOutputs: () => void;
  onDelete: () => void;
  onClick: (e: React.MouseEvent | React.TouchEvent) => void;
  isSelected?: boolean;
  disableSwipe?: boolean;
}

function TaskActionButton({
  tooltip,
  className,
  disabled = false,
  onClick,
  children,
}: {
  tooltip: string;
  className: string;
  disabled?: boolean;
  onClick?: () => void;
  children: ReactNode;
}) {
  const [tooltipVisible, setTooltipVisible] = useState(false);

  return (
    <span
      className="relative inline-flex"
      onMouseEnter={() => setTooltipVisible(true)}
      onMouseLeave={() => setTooltipVisible(false)}
      onFocus={() => setTooltipVisible(true)}
      onBlur={() => setTooltipVisible(false)}>
      <button
        type="button"
        onClick={onClick}
        className={`hcai-task-action-button ${className}`}
        disabled={disabled}
        aria-label={tooltip}>
        {children}
      </button>
      <ViewportTooltip visible={tooltipVisible} className="whitespace-nowrap">
        {tooltip}
      </ViewportTooltip>
    </span>
  );
}

export default function TaskCard({
  task,
  onReuse,
  onEditOutputs,
  onDelete,
  onClick,
  isSelected,
  disableSwipe,
}: Props) {
  const [thumbSrc, setThumbSrc] = useState<string>('');
  const [coverRatio, setCoverRatio] = useState<string>('');
  const [coverSize, setCoverSize] = useState<string>('');
  const [now, setNow] = useState(Date.now());
  const [isSwiping, setIsSwiping] = useState(false);
  const [swipeStartedSelected, setSwipeStartedSelected] = useState(false);
  const [swipeActionActive, setSwipeActionActive] = useState(false);
  const [swipeDirection, setSwipeDirection] = useState<-1 | 0 | 1>(0);
  const [streamPreviewLoaded, setStreamPreviewLoaded] = useState(false);
  const [coverShouldLoad, setCoverShouldLoad] = useState(false);
  const toggleTaskSelection = useStore((s) => s.toggleTaskSelection);
  const openFavoritePicker = useStore((s) => s.openFavoritePicker);
  const persistedStreamPreviewSrc =
    task.streamPartialImageUrls?.filter(Boolean).slice(-1)[0] || '';
  const streamPreviewSrc =
    useStore((s) => s.streamPreviews[task.id] || '') ||
    persistedStreamPreviewSrc;
  const isWaitingForStreamPreview =
    task.status === 'running' && (task.streamPartialImages || 0) > 0;
  const touchStartRef = useRef<{ x: number; y: number } | null>(null);
  const swipeResetTimerRef = useRef<number | null>(null);
  const suppressClickUntilRef = useRef(0);
  const horizontalSwipeRef = useRef(false);
  const swipeDirectionRef = useRef<-1 | 0 | 1>(0);
  const swipeActionActiveRef = useRef(false);
  const cardRef = useRef<HTMLDivElement>(null);
  const swipeOffsetRef = useRef(0);
  const pendingSwipeOffsetRef = useRef(0);
  const swipeFrameRef = useRef<number | null>(null);

  useEffect(() => {
    setCoverShouldLoad(false);
    const element = cardRef.current;
    if (!element || typeof IntersectionObserver === 'undefined') {
      setCoverShouldLoad(true);
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry?.isIntersecting) return;
        setCoverShouldLoad(true);
        observer.disconnect();
      },
      { rootMargin: '360px 0px' },
    );
    observer.observe(element);

    return () => observer.disconnect();
  }, [task.id]);

  const updateSwipeDirection = (nextDirection: -1 | 0 | 1) => {
    if (swipeDirectionRef.current === nextDirection) return;
    swipeDirectionRef.current = nextDirection;
    setSwipeDirection(nextDirection);
  };

  const updateSwipeActionActive = (nextActive: boolean) => {
    if (swipeActionActiveRef.current === nextActive) return;
    swipeActionActiveRef.current = nextActive;
    setSwipeActionActive(nextActive);
  };

  const applySwipeOffset = (offset: number) => {
    swipeOffsetRef.current = offset;
    if (cardRef.current) {
      cardRef.current.style.transform = offset ? `translateX(${offset}px)` : '';
    }
  };

  const cancelSwipeFrame = () => {
    if (swipeFrameRef.current != null) {
      window.cancelAnimationFrame(swipeFrameRef.current);
      swipeFrameRef.current = null;
    }
  };

  const scheduleSwipeOffset = (offset: number) => {
    if (swipeFrameRef.current == null && swipeOffsetRef.current === offset)
      return;
    pendingSwipeOffsetRef.current = offset;
    if (swipeFrameRef.current != null) return;
    swipeFrameRef.current = window.requestAnimationFrame(() => {
      swipeFrameRef.current = null;
      applySwipeOffset(pendingSwipeOffsetRef.current);
    });
  };

  const isTagScrollTarget = (target: EventTarget | null) => {
    return (
      target instanceof Element &&
      Boolean(target.closest('[data-tag-scroll-area]'))
    );
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    if (disableSwipe || isTagScrollTarget(e.target)) {
      touchStartRef.current = null;
      horizontalSwipeRef.current = false;
      setIsSwiping(false);
      cancelSwipeFrame();
      applySwipeOffset(0);
      updateSwipeDirection(0);
      updateSwipeActionActive(false);
      return;
    }

    if (swipeResetTimerRef.current != null) {
      window.clearTimeout(swipeResetTimerRef.current);
      swipeResetTimerRef.current = null;
    }
    touchStartRef.current = {
      x: e.touches[0].clientX,
      y: e.touches[0].clientY,
    };
    horizontalSwipeRef.current = false;
    setSwipeStartedSelected(Boolean(isSelected));
    updateSwipeActionActive(false);
    updateSwipeDirection(0);
    cancelSwipeFrame();
    applySwipeOffset(0);
    setIsSwiping(true);
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (isTagScrollTarget(e.target)) return;
    if (!touchStartRef.current) return;
    const deltaX = e.touches[0].clientX - touchStartRef.current.x;
    const deltaY = e.touches[0].clientY - touchStartRef.current.y;

    // 如果主要是水平滑动
    if (Math.abs(deltaX) > Math.abs(deltaY) && Math.abs(deltaX) > 10) {
      horizontalSwipeRef.current = true;
      e.preventDefault();
      // 限制滑动距离，例如最大 60px
      const boundedOffset = Math.max(-60, Math.min(60, deltaX));
      const nextDirection = boundedOffset > 0 ? 1 : boundedOffset < 0 ? -1 : 0;
      const nextActionActive = Math.abs(deltaX) >= 40;
      scheduleSwipeOffset(boundedOffset);
      updateSwipeDirection(nextDirection);
      updateSwipeActionActive(nextActionActive);
    }
  };

  const handleTouchEnd = (e: React.TouchEvent) => {
    if (isTagScrollTarget(e.target)) {
      touchStartRef.current = null;
      horizontalSwipeRef.current = false;
      setIsSwiping(false);
      cancelSwipeFrame();
      updateSwipeDirection(0);
      updateSwipeActionActive(false);
      return;
    }

    setIsSwiping(false);
    cancelSwipeFrame();
    updateSwipeDirection(0);

    if (!touchStartRef.current) return;
    const deltaX = e.changedTouches[0].clientX - touchStartRef.current.x;
    touchStartRef.current = null;
    const isSwipeAction = horizontalSwipeRef.current && Math.abs(deltaX) > 40;
    horizontalSwipeRef.current = false;
    updateSwipeActionActive(isSwipeAction);
    swipeResetTimerRef.current = window.setTimeout(() => {
      updateSwipeActionActive(false);
      swipeResetTimerRef.current = null;
    }, 220);

    // 如果是水平滑动，且垂直偏移较小，认为是滑动选择
    if (isSwipeAction) {
      suppressClickUntilRef.current = Date.now() + 350;
      e.preventDefault();
      e.stopPropagation();
      toggleTaskSelection(task.id);
    }
  };

  const handleTouchCancel = () => {
    touchStartRef.current = null;
    horizontalSwipeRef.current = false;
    setIsSwiping(false);
    cancelSwipeFrame();
    updateSwipeDirection(0);
    updateSwipeActionActive(false);
  };

  useEffect(
    () => () => {
      if (swipeResetTimerRef.current != null) {
        window.clearTimeout(swipeResetTimerRef.current);
      }
      cancelSwipeFrame();
    },
    [],
  );

  useEffect(() => {
    if (!isSwiping) {
      applySwipeOffset(0);
    }
  }, [isSwiping]);

  useEffect(() => {
    setStreamPreviewLoaded(false);
  }, [streamPreviewSrc, task.id]);

  // 定时更新运行中任务的计时
  useEffect(() => {
    if (
      task.status !== 'running' &&
      !(
        task.status === 'error' &&
        (task.falRecoverable || task.customRecoverable)
      )
    )
      return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    setNow(Date.now());
    return () => clearInterval(id);
  }, [task.customRecoverable, task.falRecoverable, task.status]);

  // 加载缩略图
  useEffect(() => {
    setCoverRatio('');
    setCoverSize('');
    setThumbSrc('');

    let cancelled = false;
    const remoteImageUrl = task.rawImageUrls?.[0] || '';
    const imageId = task.outputImages?.[0];
    let unsubscribe: (() => void) | undefined;

    if (task.status === 'done' && !coverShouldLoad) {
      return () => {
        cancelled = true;
      };
    }

    if (remoteImageUrl) {
      setThumbSrc(remoteImageUrl);
      return () => {
        cancelled = true;
      };
    }

    const applyThumbnail = (thumbnail: {
      dataUrl: string;
      width?: number;
      height?: number;
    }) => {
      if (cancelled) return;
      setThumbSrc(thumbnail.dataUrl);
      if (thumbnail.width && thumbnail.height) {
        setCoverRatio(formatImageRatio(thumbnail.width, thumbnail.height));
        setCoverSize(`${thumbnail.width}×${thumbnail.height}`);
      }
    };

    if (imageId) {
      unsubscribe = subscribeImageThumbnail(imageId, applyThumbnail);
      ensureImageThumbnailCached(imageId)
        .then((thumbnail) => {
          if (cancelled || !thumbnail) return;
          applyThumbnail(thumbnail);
        })
        .catch(() => {
          if (!cancelled) setThumbSrc('');
        });
    }

    return () => {
      cancelled = true;
      unsubscribe?.();
    };
  }, [coverShouldLoad, task.outputImages, task.rawImageUrls, task.status]);

  const duration = (() => {
    let seconds: number;
    if (
      task.status === 'running' ||
      task.falRecoverable ||
      task.customRecoverable
    ) {
      seconds = Math.floor((now - task.createdAt) / 1000);
    } else if (task.elapsed != null) {
      seconds = Math.floor(task.elapsed / 1000);
    } else {
      return '00:00';
    }
    const mm = String(Math.floor(seconds / 60)).padStart(2, '0');
    const ss = String(seconds % 60).padStart(2, '0');
    return `${mm}:${ss}`;
  })();
  const showSwipeAction = swipeActionActive;
  const isFalReconnecting = task.status === 'error' && task.falRecoverable;
  const isCustomReconnecting =
    task.status === 'error' && task.customRecoverable;
  const showRunningTimer =
    task.status === 'running' || isFalReconnecting || isCustomReconnecting;
  const fallbackCoverSize = task.params.size.replace(/x/g, '×');
  const fallbackCoverRatio = (() => {
    const match = task.params.size.match(/^\s*(\d+)\s*[xX×]\s*(\d+)\s*$/);
    if (!match) return '';
    return formatImageRatio(Number(match[1]), Number(match[2]));
  })();
  const displayCoverRatio = coverRatio || fallbackCoverRatio;
  const displayCoverSize = coverSize || fallbackCoverSize;
  const swipeBgClass = showSwipeAction
    ? swipeStartedSelected
      ? 'bg-gray-500 dark:bg-gray-600'
      : 'bg-blue-500'
    : 'bg-gray-200 dark:bg-gray-700';

  const isAgentTask =
    task.sourceMode === 'agent' ||
    Boolean(task.agentConversationId || task.agentRoundId);
  const showPendingPrompt = isAgentTaskPromptPending(task);
  const outputImageCount = Math.max(
    task.outputImages?.length || 0,
    task.rawImageUrls?.length || 0,
  );
  const modelDisplay =
    task.apiModel || task.apiProfileName || task.apiProvider || '';
  const isInterrupted =
    task.status === 'error' && task.error === '已停止生成。';

  return (
    <div className="relative rounded-xl">
      {/* 侧滑底图 */}
      <div
        className={`absolute inset-0 rounded-xl flex items-center transition-opacity duration-200 pointer-events-none ${
          isSwiping || swipeDirection !== 0 || swipeActionActive
            ? 'opacity-100'
            : 'opacity-0'
        } ${swipeBgClass} ${
          swipeDirection > 0 ? 'justify-start pl-6' : 'justify-end pr-6'
        }`}>
        <svg
          className={`w-8 h-8 transition-transform duration-150 ${showSwipeAction ? 'scale-110 text-white' : 'scale-90 text-white/60'}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24">
          {swipeStartedSelected && showSwipeAction ? (
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          ) : (
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={3}
              d="M5 13l4 4L19 7"
            />
          )}
        </svg>
      </div>

      <div
        ref={cardRef}
        className={`hcai-task-card relative bg-white dark:bg-gray-900 rounded-xl border overflow-hidden cursor-pointer touch-pan-y will-change-transform duration-200 hover:shadow-lg dark:hover:bg-gray-800/80 ${
          isSwiping ? '!bg-white dark:!bg-gray-900' : ''
        } ${
          !isSwiping
            ? 'transition-[box-shadow,border-color,background-color,transform]'
            : 'transition-[box-shadow,border-color,background-color]'
        } ${
          task.status === 'running'
            ? 'border-blue-400 generating'
            : isSelected
              ? 'border-blue-500 shadow-md ring-2 ring-blue-500/50'
              : 'border-gray-200 dark:border-white/[0.08] hover:border-gray-300 dark:hover:border-white/[0.18]'
        }`}
        onClick={(e) => {
          if (Date.now() < suppressClickUntilRef.current) {
            e.preventDefault();
            e.stopPropagation();
            return;
          }
          onClick(e);
        }}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
        onTouchCancel={handleTouchCancel}
        draggable={task.status === 'done' && task.outputImages?.length > 0}
        onDragStart={(e) => {
          if (task.status !== 'done' || !task.outputImages?.length) return;
          const imageIds = task.outputImages;
          e.dataTransfer.setData(
            'text/plain',
            `agent-images:${imageIds.join(',')}`,
          );
          e.dataTransfer.effectAllowed = 'copy';
          // Optionally set drag image if we have thumbSrc
          if (thumbSrc) {
            const preview = document.createElement('div');
            preview.style.cssText =
              'position:fixed;left:-1000px;top:-1000px;width:100px;height:100px;border-radius:12px;overflow:hidden;box-shadow:0 4px 12px rgba(0,0,0,0.25);';
            const previewImg = document.createElement('img');
            previewImg.src = thumbSrc;
            previewImg.style.cssText =
              'width:100px;height:100px;object-fit:cover;display:block;';
            preview.appendChild(previewImg);
            document.body.appendChild(preview);
            e.dataTransfer.setDragImage(preview, 50, 50);
            setTimeout(() => preview.remove(), 0);
          }
        }}>
        {/* 选中时的角标 */}
        {isSelected && (
          <div className="absolute top-2 right-2 z-10 w-5 h-5 bg-blue-500 rounded-full flex items-center justify-center shadow-sm">
            <svg
              className="w-3 h-3 text-white"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={3}
                d="M5 13l4 4L19 7"
              />
            </svg>
          </div>
        )}
        <div className="flex h-40">
          {/* 左侧图片区域 */}
          <div className="w-40 min-w-[10rem] h-full bg-gray-100 dark:bg-black/20 relative flex items-center justify-center overflow-hidden flex-shrink-0">
            {task.status === 'running' && streamPreviewSrc && (
              <>
                <img
                  src={streamPreviewSrc}
                  className={`h-full w-full object-cover ${streamPreviewLoaded ? '' : 'hidden'}`}
                  alt=""
                  onLoad={() => setStreamPreviewLoaded(true)}
                  onError={() => setStreamPreviewLoaded(false)}
                />
                <span className="hcai-stream-preview-badge absolute top-1.5 right-1.5 flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-medium text-white backdrop-blur-sm sm:text-xs">
                  流式预览
                </span>
              </>
            )}
            {task.status === 'running' &&
              (!streamPreviewSrc || !streamPreviewLoaded) && (
                <div className="flex flex-col items-center gap-2">
                  <svg
                    className="w-8 h-8 text-blue-400 animate-spin"
                    fill="none"
                    viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                    />
                  </svg>
                  <span className="text-xs text-gray-400 dark:text-gray-500">
                    {isWaitingForStreamPreview
                      ? '等待流式预览...'
                      : '生成中...'}
                  </span>
                </div>
              )}
            {task.status === 'error' && isFalReconnecting && (
              <div className="flex flex-col items-center gap-1 px-2">
                <svg
                  className="w-7 h-7 text-yellow-400"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                  />
                </svg>
                <span className="text-xs text-yellow-500 text-center leading-tight">
                  重连中
                </span>
              </div>
            )}
            {task.status === 'error' && !isFalReconnecting && (
              <div className="flex flex-col items-center gap-1 px-2">
                <svg
                  className={`w-7 h-7 ${isInterrupted ? 'text-yellow-400' : 'text-red-400'}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 9v2m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <span
                  className={`text-xs text-center leading-tight ${isInterrupted ? 'text-yellow-500' : 'text-red-400'}`}>
                  {isInterrupted ? '已停止' : '失败'}
                </span>
              </div>
            )}
            {task.status === 'done' && thumbSrc && (
              <>
                <img
                  src={thumbSrc}
                  data-image-id={task.outputImages[0] || undefined}
                  data-output-image-ids={
                    task.outputImages.length
                      ? task.outputImages.join(',')
                      : undefined
                  }
                  className="saveable-image w-full h-full object-cover"
                  loading="lazy"
                  onLoad={(e) => {
                    if (!task.rawImageUrls?.[0]) return;
                    const image = e.currentTarget;
                    if (image.naturalWidth > 0 && image.naturalHeight > 0) {
                      setCoverRatio(
                        formatImageRatio(
                          image.naturalWidth,
                          image.naturalHeight,
                        ),
                      );
                      setCoverSize(
                        `${image.naturalWidth}×${image.naturalHeight}`,
                      );
                    }
                  }}
                  alt=""
                />
                {outputImageCount > 1 && (
                  <span className="absolute bottom-1 right-1 bg-black/60 text-white text-xs px-1.5 py-0.5 rounded">
                    {outputImageCount}
                  </span>
                )}
              </>
            )}
            {task.status === 'done' && !thumbSrc && (
              <svg
                className="w-8 h-8 text-gray-300"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
            )}
            {/* 运行中显示耗时，完成后显示封面图比例与分辨率标签 */}
            <div className="absolute top-1.5 left-1.5 flex items-center gap-1">
              {showRunningTimer || task.status !== 'done' ? (
                <span className="flex items-center gap-1 bg-black/50 text-white text-[10px] sm:text-xs px-1.5 py-0.5 rounded backdrop-blur-sm font-mono">
                  <svg
                    className="w-3 h-3"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                  {duration}
                </span>
              ) : displayCoverRatio && displayCoverSize ? (
                <>
                  <span className="bg-black/50 text-white text-[10px] sm:text-xs px-1.5 py-0.5 rounded backdrop-blur-sm font-mono">
                    {displayCoverRatio}
                  </span>
                  <span className="bg-black/50 text-white/90 text-[10px] sm:text-xs px-1.5 py-0.5 rounded backdrop-blur-sm font-medium">
                    {displayCoverSize}
                  </span>
                </>
              ) : null}
            </div>
          </div>

          {/* 右侧信息区域 */}
          <div className="flex-1 p-3 flex flex-col min-w-0">
            <div className="flex-1 min-h-0 mb-2 overflow-hidden">
              {showPendingPrompt ? (
                <div className="leading-relaxed">
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    正在生成……
                  </p>
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                    输入内容将在响应完成时接收
                  </p>
                </div>
              ) : (
                <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed line-clamp-3">
                  {task.prompt || '(无提示词)'}
                </p>
              )}
            </div>
            <div className="mt-auto flex flex-col gap-1.5">
              {/* 参数与信息：横向滚动 */}
              <div
                data-tag-scroll-area
                className="flex overflow-x-auto hide-scrollbar pt-0.5 gap-1.5 whitespace-nowrap mask-edge-r min-w-0 pr-2"
                onTouchStart={(e) => e.stopPropagation()}
                onTouchMove={(e) => e.stopPropagation()}
                onTouchEnd={(e) => e.stopPropagation()}
                onTouchCancel={(e) => e.stopPropagation()}>
                {modelDisplay && (
                  <span
                    className="hcai-task-model-pill flex items-center gap-1 px-1.5 py-0.5 rounded bg-gray-100 dark:bg-white/[0.04] text-gray-600 dark:text-gray-300 text-xs flex-shrink-0"
                    title={modelDisplay}>
                    <span className="truncate max-w-[8rem]">
                      {modelDisplay}
                    </span>
                  </span>
                )}
              </div>
              {/* 操作按钮 */}
              <div
                data-tag-scroll-area
                className="hcai-task-actions flex items-center gap-0.5 flex-shrink-0 mt-0.5 max-w-full overflow-visible"
                onClick={(e) => e.stopPropagation()}
                onTouchStart={(e) => e.stopPropagation()}
                onTouchMove={(e) => e.stopPropagation()}
                onTouchEnd={(e) => e.stopPropagation()}
                onTouchCancel={(e) => e.stopPropagation()}>
                {task.status === 'error' && !isFalReconnecting && (
                  <TaskActionButton
                    tooltip="重试任务"
                    onClick={() => retryTask(task)}
                    className="p-1.5 rounded-md hover:bg-blue-50 dark:hover:bg-blue-950/30 text-gray-400 hover:text-blue-500 transition">
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                      />
                    </svg>
                  </TaskActionButton>
                )}
                <TaskActionButton
                  tooltip={task.isFavorite ? '编辑收藏夹' : '收藏任务'}
                  onClick={() => openFavoritePicker([task.id])}
                  className={`p-1.5 rounded-md transition ${
                    task.isFavorite
                      ? 'text-yellow-400 hover:bg-yellow-50 dark:hover:bg-yellow-500/10'
                      : 'text-gray-400 hover:text-yellow-400 hover:bg-yellow-50 dark:hover:bg-yellow-500/10'
                  }`}>
                  <svg
                    className="w-4 h-4"
                    fill={task.isFavorite ? 'currentColor' : 'none'}
                    stroke="currentColor"
                    viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"
                    />
                  </svg>
                </TaskActionButton>
                <TaskActionButton
                  tooltip="复用配置"
                  onClick={onReuse}
                  className="p-1.5 rounded-md hover:bg-blue-50 dark:hover:bg-blue-950/30 text-gray-400 hover:text-blue-500 transition">
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6"
                    />
                  </svg>
                </TaskActionButton>
                <TaskActionButton
                  tooltip="编辑输出"
                  onClick={onEditOutputs}
                  className="p-1.5 rounded-md hover:bg-green-50 dark:hover:bg-green-950/30 text-gray-400 hover:text-green-500 transition disabled:opacity-30"
                  disabled={outputImageCount === 0}>
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                    />
                  </svg>
                </TaskActionButton>
                <TaskActionButton
                  tooltip="删除任务"
                  onClick={onDelete}
                  className="p-1.5 rounded-md hover:bg-red-50 dark:hover:bg-red-950/30 text-gray-400 hover:text-red-500 transition">
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24">
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                  </svg>
                </TaskActionButton>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
