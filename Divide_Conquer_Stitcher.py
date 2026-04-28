import cv2
import numpy as np


def get_cosine_alpha_mask(height, width, overlap_width):
    alpha = np.ones((height, width, 3), dtype=np.float32)

    # 겹치는 구간에 대해 코사인 기반 가중치 계산 (1.0 -> 0.0)
    t = np.linspace(0, np.pi, overlap_width)
    cosine_gradient = (np.cos(t) + 1) / 2

    for i in range(overlap_width):
        alpha[:, i] = cosine_gradient[i]

    return alpha


def stitch_two_images(img_left, img_right):
    if img_left is None:
        return img_right
    if img_right is None:
        return img_left

    # 1. 특징점 검출 (SIFT)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_left, None)
    kp2, des2 = sift.detectAndCompute(img_right, None)

    # 2. 특징점 매칭 (FLANN)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]

    # 매칭 실패 시 더 큰 이미지를 반환하거나 원본 유지
    if len(good_matches) < 4:
        return img_left if img_left.shape[1] > img_right.shape[1] else img_right

    # 3. Homography 계산 (RANSAC)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    # 4. Warping 및 결과 캔버스 생성
    h1, w1 = img_left.shape[:2]
    h2, w2 = img_right.shape[:2]

    # 두 이미지가 합쳐질 넉넉한 크기의 결과창
    warped_right = cv2.warpPerspective(img_right, H, (w1 + w2, h1))

    # 5. 비선형 코사인 블렌딩
    # 겹치는 영역(overlap)을 w1(왼쪽 이미지의 너비) 전체로 설정하여 부드럽게 합성
    overlap_width = w1
    stitched = warped_right.astype(np.float32)

    alpha_mask = get_cosine_alpha_mask(h1, overlap_width, overlap_width)

    # 영역 합치기: (왼쪽 * 가중치) + (오른쪽 * (1-가중치))
    stitched[:, :overlap_width] = img_left.astype(
        np.float32
    ) * alpha_mask + warped_right[:, :overlap_width].astype(np.float32) * (
        1 - alpha_mask
    )

    return np.clip(stitched, 0, 255).astype(np.uint8)


def divide_and_conquer_stitch(image_list):
    n = len(image_list)

    # Base Case
    if n == 1:
        return image_list[0]
    if n == 2:
        return stitch_two_images(image_list[0], image_list[1])

    # Divide: 반으로 나누기
    mid = n // 2
    left_side = divide_and_conquer_stitch(image_list[:mid])
    right_side = divide_and_conquer_stitch(image_list[mid:])

    # Conquer: 합치기
    return stitch_two_images(left_side, right_side)


def main():
    # 이미지 경로 리스트
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
    images = [cv2.imread(p) for p in image_paths]

    if any(img is None for img in images):
        print("에러: 이미지 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    print(f"총 {len(images)}장의 이미지를 분할 정복 방식으로 합성을 시작합니다...")

    # 최종 파노라마 생성
    result = divide_and_conquer_stitch(images)

    # 결과 출력 및 저장
    cv2.imshow("D&C Cosine Panorama", result)
    cv2.imwrite("final_panorama.jpg", result)
    print("합성 완료: final_panorama.jpg로 저장되었습니다.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
